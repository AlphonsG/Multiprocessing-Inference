from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from multiprocessing.managers import BaseManager
from threading import Lock, Semaphore, Thread
from typing import Any

import torch
from dotenv import load_dotenv
from torch.multiprocessing import (Event, Manager, Process, Queue,
                                   current_process, set_start_method)

HOST_ENV_VAR = 'MODEL_MANAGER_HOST'
PORT_ENV_VAR = 'MODEL_MANAGER_PORT'
MANAGER_AUTH_KEY_ENV_VAR = 'MODEL_MANAGER_AUTH_KEY'
PROCESS_AUTH_KEY_ENV_VAR = 'PROCESS_AUTH_KEY'
_MODEL_PROXY_SENTINEL = 'STOP_FROM_WORKER'


class QueueManager(BaseManager):
    pass


class ModelManager:  # TODO for modelproxy, what if people still want original attributes?
    """Manager for machine learning models. from torch.nn import Module

    Initializes a CPU or CUDA based optical flow model (class that inherits
    rainbow.optical_flow.base_model.BaseModel) and provides a proxy to it.
    The proxy has the same functionality as the original model but instances of
    the proxy can now be passed to separate Python worker processes, e.g.
    without pickling errors, while allowing the single optical flow model
    instance to be shared between them, e.g. without duplicating the model in
    memory. Solves issues such as (1)
    https://github.com/pytorch/pytorch/issues/16943#issuecomment-462956544 and
    (2)
    https://github.com/pytorch/pytorch/issues/35472#issuecomment-879141708.
    Additionally, automatically manages the lifecycle of the model, such as
    batch image inference from multiple worker processes and model termination.
    """
    _SENTINEL = 'STOP'

    def __init__(
        self, model: Model,
        max_num_sim_jobs: int = 1,
        timeout: int | None = None,
    ) -> None:
        """Initializes the model manager.

        When used, ModelManager must be guarded with an
        if __name__ == '__main__' guard.

        Args:
            model: A Model object.
            max_num_sim_jobs: The maximum number of inference jobs that can
                be fed to the model simultaneously. Greater values result in
                higher system/GPU memory usage.
            timeout: The maximum amount of time, in seconds, to wait for
                jobs being processed by the model to finish if execution leaves
                the ModelManager context before they are completed. If none,
                waits indefinitely.
        """
        set_start_method('spawn', force=True)  # for CUDA runtime support
        self._max_num_sim_jobs = max_num_sim_jobs
        self._model = model
        self._timeout = timeout

    def __enter__(self) -> ModelProxy:
        # Queues to allow processes to communicate with the model manager
        self._to_model_runner, self._from_model_runner = Queue(), Queue()

        # start a QueueManager that will allow processes to send/receive
        # data to/from the process with the model (via queues)
        load_dotenv()
        try:
            self._host = os.environ[HOST_ENV_VAR]
            self._port = int(os.environ[PORT_ENV_VAR])
            self._manager_auth_key = os.environ[MANAGER_AUTH_KEY_ENV_VAR]
            self._process_auth_key = os.environ[PROCESS_AUTH_KEY_ENV_VAR]
        except (KeyError, ValueError) as e:
            msg = ('ModelManager environment variables not correctly set.')
            raise ValueError(msg) from e

        current_process().authkey = bytes(self._process_auth_key,
                                          encoding='utf8')
        self._event = Event()
        queue_manager = Process(target=self._start_queue_manager)
        queue_manager.start()

        # start the process that will load and run the model
        self._model_runner = Process(target=self._run_model)
        self._model_runner.start()
        self._queue_manager = queue_manager
        self._event.wait()  # wait for QueueManager to be ready

        return ModelProxy()

    def _start_queue_manager(self) -> None:
        QueueManager.register('to_model_runner',
                              callable=lambda: self._to_model_runner)
        QueueManager.register('from_model_runner',
                              callable=lambda: self._from_model_runner)
        queue_manager = QueueManager(
            address=(self._host, self._port), authkey=bytes(
                self._manager_auth_key, encoding='utf8'))
        # manager.start() fails with pickling errors, hence why using blocking
        # server in this other process
        server = queue_manager.get_server()
        self._event.set()
        server.serve_forever()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        # signal threads using the model to terminate once finished processing
        # any current jobs, if still running after timeout then force
        # termination
        self._to_model_runner.put(self._SENTINEL)
        self._model_runner.join(self._timeout)
        if self._model_runner.is_alive():
            self._model_runner.terminate()
        self._queue_manager.terminate()
        self._queue_manager.join()

    def _run_model(self) -> None:
        # TODO timeout worker threads?
        self._threads = []
        self._thread_queues = []
        self._thread_queues_lock, self._threads_lock = Lock(), Lock()
        self._semaphore = Semaphore(self._max_num_sim_jobs)
        self._model.load()
        while self._to_model_runner.get() != self._SENTINEL:
            # external process using model - create runner thread
            # P.S. plain Queue() causes inheritance error
            to_thread = Manager().Queue()
            from_thread = Manager().Queue()
            thread = Thread(target=self._model_runner_thread,
                            args=(to_thread, from_thread))
            thread.start()
            self._from_model_runner.put((to_thread, from_thread))
            self._threads.append(thread)
            self._thread_queues.append(to_thread)

        # signal threads to terminate
        with self._thread_queues_lock:
            for to_thread in self._thread_queues:
                to_thread.put(self._SENTINEL)

        # prevent race condition in the case of a thread removing itself from
        # thread list due to external termination request already in progress
        with self._threads_lock:
            threads = self._threads.copy()

        for thread in threads:  # wait for threads to end
            thread.join()

    def _model_runner_thread(
        self,
        predict: Queue,
        predicted: Queue,
    ) -> None:
        while isinstance((rcvd := predict.get()), torch.Tensor):
            self._semaphore.acquire()
            preds = self._model.predict(rcvd).cpu()
            preds.share_memory_()
            self._semaphore.release()
            predicted.put(preds)

        # no more jobs from external process
        # in this case, stop parent process keeping track of thread
        if rcvd == _MODEL_PROXY_SENTINEL:
            with self._thread_queues_lock:
                i = self._thread_queues.index(predict)
                self._thread_queues.remove(predict)
                with self._threads_lock:
                    self._threads.remove(self._threads[i])


class ModelProxy:
    """Proxy for a Model managed by ModelManager."""

    def __init__(self) -> None:
        """Inits ModelProxy by preparing for a connection to ModelManager."""
        load_dotenv()
        try:
            self._host = os.environ[HOST_ENV_VAR]
            self._port = int(os.environ[PORT_ENV_VAR])
            self._manager_auth_key = os.environ[MANAGER_AUTH_KEY_ENV_VAR]
            self._process_auth_key = os.environ[PROCESS_AUTH_KEY_ENV_VAR]
        except (KeyError, ValueError) as e:
            msg = 'ModelManager environment variables not correctly set.'
            raise ValueError(msg) from e

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Proxy method for the '.predict()' method of the Model.
        """
        data.share_memory_()
        data = data.cpu()  # sharing CUDA tensor can cause RuntimeError
        self._predict.put(data)  # share tensor with process managing the model

        return self._predicted.get()  # get inference results, again shared

    def __enter__(self) -> None:
        current_process().authkey = bytes(self._process_auth_key,
                                          encoding='utf8')
        QueueManager.register('to_model_runner')
        QueueManager.register('from_model_runner')
        self._remote_manager = QueueManager(address=(
            self._host, self._port), authkey=bytes(
                self._manager_auth_key, encoding='utf8'))
        self._remote_manager.connect()
        self._remote_manager.to_model_runner().put(None)
        self._predict, self._predicted = (
            self._remote_manager.from_model_runner().get())

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        # signal that no more jobs will be sent to the model manager
        self._predict.put(_MODEL_PROXY_SENTINEL)


class Model:
    """Interface for machine learning models."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self) -> None:
        """Loads a machine learning model to system or GPU memory.

        For example, this could

        Raises:
            NotImplementedError: Method not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Makes predictions against input data using the loaded model.

        For optimal performance, minimize python byte code here and try having
        only CUDA blah, blah e.g. nn.forward post-preposti

        Args:
            data: The input data.

        Returns:
            The predictions made by the model.

        Raises:
            NotImplementedError: Method not implemented.
        """
        raise NotImplementedError
