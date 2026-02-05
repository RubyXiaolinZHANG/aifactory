import heapq
from copy import deepcopy


class BoundedMaxHeapDictList:
    """
    Maintains a fixed-size list of dictionaries, sorted by a specified key, retaining the largest K items.
    Implemented using heapq (min-heap) for higher efficiency.
    """
    def __init__(self, max_size, sort_key):
        self.max_size = max_size
        self.sort_key = sort_key
        # Internally maintain a min-heap, heap elements are (sort_key_value, original_dictionary)
        self._heap = []

    def insert(self, item_dict):
        if not isinstance(item_dict, dict) or self.sort_key not in item_dict:
            raise ValueError(f"The item to insert must be a dictionary containing the key '{self.sort_key}'")

        key_value = item_dict[self.sort_key]
        heap_item = (key_value, item_dict)

        if len(self._heap) < self.max_size:
            # Heap not full, push directly
            heapq.heappush(self._heap, heap_item)
        else:
            # Heap full, compare new value with the current minimum (heap[0])
            if key_value > self._heap[0][0]:
                # New value larger than current minimum, replace the minimum
                head = self._heap[0][1]
                heapq.heapreplace(self._heap, heap_item)
                return head
            else:
                return None
            # Else: new value <= current minimum, discard directly

    def get_sorted_items(self):
        """Return the original dictionary list sorted in ascending order by the sort key."""
        # Note: The heap itself is not fully sorted, only heap-ordered. Sort the tuples by key.
        sorted_heap = sorted(self._heap)  # Sort by the first element of the tuple (key value)
        return [item[1] for item in sorted_heap]

    def get_min_value(self):
        """Get the minimum key value in the current heap (if heap is not empty)."""
        return self._heap[0][0] if self._heap else None

    def __repr__(self):
        return f"BoundedMaxHeapDictList(max_size={self.max_size}, sort_key='{self.sort_key}', items={self.get_sorted_items()})"
