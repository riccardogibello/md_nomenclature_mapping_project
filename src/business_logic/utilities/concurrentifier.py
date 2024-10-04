import json
from typing import Any, Callable, Union, Sized, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
import concurrent.futures

# This general formula can be used to estimate the required parameters, in case not all the values are provided
# by the system information: number virtual processors = MAX_PROCESSES * MAX_THREADS
MAX_PROCESSES = 4
MAX_THREADS = 1


def _divide_into_batches(
        sized_element: np.ndarray | list | DataFrame | dict,
        batch_size: int,
) -> list[Any]:
    """
    This method divides the input element in batches of batch size elements and returns a list containing the batches
    as elements.

    :param sized_element: The Sized element to be divided in batches.
    :param batch_size: The number of elements in each batch.

    :return: A list containing the batches as elements.
    """
    # Create a list of the final batches, that will contain the batches of the dataframe/sized element
    final_batches = []
    start_index: Any = 0
    end_index: Any = len(sized_element)
    step: Any = batch_size
    # The number of elements in each batch
    if type(sized_element) is pd.DataFrame:
        # Divide the DataFrame in lists of final_batch_size elements
        for i in range(start_index, end_index, step):
            final_batches.append(sized_element.iloc[i:i + batch_size])
    elif type(sized_element) is list or type(sized_element) is np.ndarray:
        # Divide the Sized element in lists of final_batch_size elements
        for i in range(start_index, end_index, step):
            final_batches.append(sized_element[i:i + batch_size])
    elif type(sized_element) is dict:
        # Divide the dictionary into batches that contain the same amount of
        # key-value pairs
        keys = list(sized_element.keys())
        end_index = len(keys)
        values = list(sized_element.values())
        for i in range(start_index, end_index, step):
            i: Any
            tmp_map = {}
            for j in range(i, min(i + batch_size, len(keys))):
                j: Any
                tmp_map[keys[j]] = values[j]
            final_batches.append(tmp_map)
    else:
        raise ValueError("The input element is not supported.")

    return final_batches


def _wait_for_generic_future_results(
        wait_all: bool,
        future_list: list,
        old_submitted_task: int,
) -> Tuple[Any, int]:
    """
    This method waits for the results of the threads/processes and returns the results in a list. Moreover, it returns
    the updated number of submitted tasks to the threads/processes, considering the ones that returned the results.

    :param wait_all: This indicates whether the method must wait for all the threads/processes to finish.
    :param future_list: The list of the futures that have been submitted to the threads/processes.
    :param old_submitted_task: The current number of submitted tasks to the threads/processes.

    :return: A list containing the results given by the threads/processes and the updated number of submitted tasks.
    """
    tmp_returned_future = []
    for future in concurrent.futures.as_completed(future_list):
        # when the future is done, remove it from the list
        future_list.remove(future)

        # Get back from the thread/process a JSON string, which contains the results in a list
        tmp_result_string: Optional[str] = future.result()
        if tmp_result_string is not None:
            # Transform the JSON string in a list
            tmp_result = json.loads(tmp_result_string)
            if type(tmp_result) is not list:
                tmp_result = [tmp_result]
            # Extend the returned values list with the results given by the thread/process
            tmp_returned_future.extend(tmp_result)

        # Subtract 1 to the number of submitted tasks to the threads/processes
        old_submitted_task = old_submitted_task - 1

        # If it is not necessary to wait all the results,
        if not wait_all:
            # Then break
            break

    return tmp_returned_future, old_submitted_task


def _run_in_parallel(
        max_number: int,
        batch: Union[Sized, pd.DataFrame],
        additional_batch_callback_parameters: Any,
        executor_creation_callback: Callable[[int], Any],
        processing_batch_callback: Callable[..., Any],
        max_number_threads: Optional[int] = None,
) -> str:
    """
    This method runs the processing_batch_callback in parallel over multiple threads/processes. The maximum number of
    threads/processes is given by the max_number parameter. The batch is divided in sub-batches, so that each thread
    receives a sub-batch to process. The additional_batch_callback_parameters are passed to the
    processing_batch_callback. The executor_creation_callback is used to create the executor, which can be a
    ThreadPoolExecutor or a ProcessPoolExecutor. The max_number_threads parameter is not None when the
    executor_creation_callback returns a ProcessPoolExecutor (i.e., the execution is parallelized over
    multiple processes).

    :param max_number: The maximum number of threads/processes.
    :param batch: The Sized element to be processed.
    :param additional_batch_callback_parameters: The additional parameters to be passed to the
    processing_batch_callback.
    :param executor_creation_callback: The callback to create the executor.
    :param max_number_threads: The maximum number of threads.
    :param processing_batch_callback: The callback to process the batch.

    :return: A JSON string which contains all the results fetched from the different threads/processes.
    """
    # Create a list in which the results from the single threads/processes are stored
    returned_results: list[np.ndarray[Any]] = []

    # If the current execution is single-threaded/single-processed
    if max_number == 0:
        # Execute the processing_batch_callback on the entire batch and store the result in the returned_results list
        returned_results = [
            processing_batch_callback(
                batch,
                additional_batch_callback_parameters,
            )
        ]
    else:
        # Divide the current batch in sub-batches, so that the number of batches is equal to the number of
        # threads/processes
        batches = _divide_into_batches(
            sized_element=batch,
            batch_size=len(batch) // max_number + 1
        )

        # Create a list of the futures which have been submitted to the threads/processes
        future_list = []

        # Set a counter for the number of batches under processing / processed
        counter = 0
        # Set the maximum number of submitted jobs to the number of threads/processes
        max_submitted_jobs = max_number
        # Keep track of the number of submitted tasks to the threads/processes
        pending_tasks_count = 0

        with executor_creation_callback(
                max_submitted_jobs,
        ) as executor:
            # For each batch to be processed
            for batch in batches:
                # if the current executor is of type ThreadPoolExecutor,
                if type(executor) is concurrent.futures.ThreadPoolExecutor:
                    if additional_batch_callback_parameters is None:
                        # Send the batch to the executor
                        future_list.append(
                            executor.submit(
                                processing_batch_callback,
                                batch,
                            )
                        )
                    else:
                        # Send the batch to the executor
                        future_list.append(
                            executor.submit(
                                processing_batch_callback,
                                batch,
                                *additional_batch_callback_parameters,
                            )
                        )
                else:
                    # Send the batch to the executor
                    future_list.append(
                        executor.submit(
                            _run_in_parallel,
                            max_number_threads,
                            batch,
                            additional_batch_callback_parameters,
                            concurrent.futures.ThreadPoolExecutor,
                            processing_batch_callback,
                            None,
                        )
                    )

                # Add 1 to the total number of submitted tasks and still in processing
                pending_tasks_count = pending_tasks_count + 1

                is_last_batch = counter == len(batches) - 1
                # If the number of submitted tasks is equal to the number of threads/processes
                # or if the current batch is the last one
                if pending_tasks_count == max_submitted_jobs or is_last_batch:
                    # If the current batch is the last one, wait for all the threads/processes to finish
                    future_list_results, pending_tasks_count = _wait_for_generic_future_results(
                        # Wait all the threads/processes to finish if the current batch is the last one
                        wait_all=is_last_batch,
                        future_list=future_list,
                        old_submitted_task=pending_tasks_count,
                    )

                    # For each result given by the threads/processes
                    for result in future_list_results:
                        # Save the given result in the returned values list
                        returned_results.append(result)

                    del future_list_results

                # Update the counter of the processed / still in process batches
                counter = counter + 1

        # Verify that the results have been given by all the threads/processes
        assert pending_tasks_count == 0
    # Return a list that contains the returned values by the processing_batch_callback,
    # one for each batch (thread/process)
    return json.dumps(returned_results)


def perform_parallelization(
        input_sized_element: Union[Sized, DataFrame],
        processing_batch_callback: Callable[..., Any],
        additional_batch_callback_parameters: Optional[Any] = None,
        threads_number: Optional[int] = MAX_THREADS,
        processes_number: Optional[int] = MAX_PROCESSES,
) -> list[Any]:
    """
    This method performs a parallelization of the processing_batch_callback over the input_sized_element. The
    additional_batch_callback_parameters are passed to the processing_batch_callback.

    :param input_sized_element: The Sized element to be processed.
    :param processing_batch_callback: The callback to process the batch.
    :param additional_batch_callback_parameters: The additional parameters to be passed to the
    processing_batch_callback.
    :param threads_number: The number of threads to be used.
    :param processes_number: The number of processes to be used.

    :return: A list containing the results given by the processing_batch_callback.
    """
    # If the execution must be done in a single process
    if processes_number == 0:
        # Run the method to parallelize over multiple threads
        result = _run_in_parallel(
            max_number=threads_number,
            processing_batch_callback=processing_batch_callback,
            additional_batch_callback_parameters=additional_batch_callback_parameters,
            batch=input_sized_element,
            executor_creation_callback=concurrent.futures.ThreadPoolExecutor,
        )
    else:
        # Run the method to parallelize over multiple processes
        result = _run_in_parallel(
            max_number=processes_number,
            max_number_threads=threads_number,
            processing_batch_callback=processing_batch_callback,
            additional_batch_callback_parameters=additional_batch_callback_parameters,
            batch=input_sized_element,
            executor_creation_callback=concurrent.futures.ProcessPoolExecutor,
        )

    return json.loads(
        result
    )
