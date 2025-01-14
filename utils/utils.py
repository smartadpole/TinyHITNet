import time
from functools import wraps

__all__ = ['timeit']

# Decorator function to measure the time taken by a function
def timeit(time_len):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            wrapper.times.append(elapsed_time)
            if len(wrapper.times) % time_len == 0:
                average_time = sum(wrapper.times[-time_len:]) / time_len
                print(f"Average time for last {time_len} frames in {func.__name__}: {average_time:.4f} seconds")
            return result
        wrapper.times = []
        return wrapper
    return decorator