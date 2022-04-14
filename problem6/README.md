### Concurrent Hello World in C++

Prompt:
1. Start two threads, making one print “hello” twice and one print “world” twice (not like “hellohello”) using cout.
2. Build with cmake (add set(CMAKE_CXX_FLAGS -pthread) to use std::thread) and run, 
and observe that the “hello”s and “world”s sometimes are interleaved.
3. Use a mutex to control access to the resource (i.e. cout) so that hello is always printed twice in a row 
and world always printed twice in a row.

Expected Output:
```
$ ./hello_world 
hello	hello	world	world
```