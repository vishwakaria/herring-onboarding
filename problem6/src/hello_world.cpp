#include <iostream>
#include <thread>
#include <mutex>

std::mutex m;
int i=0;

void printHello() {
	m.lock();
	for (i=0; i<2; i++) {
		std::cout<<"hello\t";
	}
	m.unlock();
}

void printWorld() {
	m.lock();
 	for (i=0; i<2; i++) {
		std::cout<<"world\t";
	}
	m.unlock();
}

int main() {
	std::thread tHello(printHello);
	std::thread tWorld(printWorld);
	tHello.join();
	tWorld.join();
}

