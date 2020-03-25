
#include <random>
#include <ctime>
#include <iostream>

#if __GNUC__ >= 3
#define HOLO_LIKELY(x) __builtin_expect(!!(x), 1)
#define HOLO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#pragma message("likely was defined")
#else
#define HOLO_LIKELY(x) (x)
#define HOLO_UNLIKELY(x) (x)
#pragma message("likely was not defined")
#endif

static std::vector<int> GetOdds(const int count) {
	std::vector<int> buffer;
	buffer.reserve(count);

	while (buffer.size() < static_cast<std::size_t>(count)) {
		const int value = rand();
		if (1 == (value % 2)) {
			buffer.push_back(value);
		}
	}

	return buffer;
}

static std::vector<int> GetEvens(const int count) {
	std::vector<int> buffer;
	buffer.reserve(count);

	while (buffer.size() < static_cast<std::size_t>(count)) {
		const int value = rand();
		if (0 == (value % 2)) {
			buffer.push_back(value);
		}
	}

	return buffer;
}

int main(int argc, char* argv[]) {
	const int count = 1000000000;

	const std::vector<int> odds = GetOdds(count);
	const std::vector<int> evens = GetEvens(count);

	time_t start = 0;
	time_t end = 0;

	int output = 0;
	int other = 0;

	/// Test odds with likely
	{
		output = other = 0;
		start = time(NULL);
		for (int i = 0; i < count; ++i) {
			if (HOLO_LIKELY(odds[i]%2)) {
				++output;
			} else {
				++other;
			}
		} 
		end = time(NULL);
		std::cout << "Test odds with likely, cost=" << end-start << " output=" << output << " other=" << other << std::endl;
	}
	

	/// Test evens with likely
	{
		output = other = 0;
		start = time(NULL);
		for (int i = 0; i < count; ++i) {
			if (HOLO_UNLIKELY(evens[i]%2)) {
				++other;
			} else {
				++output;
			}
		} 
		end = time(NULL);
		std::cout << "Test odds with likely, cost=" << end-start << " output=" << output << " other=" << other << std::endl;
	}

	/// Test odds without likely
	{
		output = other = 0;
		start = time(NULL);
		for (int i = 0; i < count; ++i) {
			if (odds[i]%2) {
				++output;
			} else {
				++other;
			}
		} 
		end = time(NULL);
		std::cout << "Test odds without likely, cost=" << end-start << " output=" << output << " other=" << other << std::endl;
	}

	/// Test evens without likely
	{
		output = other = 0;
		start = time(NULL);
		for (int i = 0; i < count; ++i) {
			if (evens[i]%2) {
				++other;
			} else {
				++output;
			}
		} 
		end = time(NULL);
		std::cout << "Test evens without likely, cost=" << end-start << " output=" << output << " other=" << other << std::endl;
	}

	return 0;
}

/**
 * Env: MacBook Pro 10.13.4 2.3 GHz Intel Core i5 8 GB 2133 MHz LPDDR3
 * |           Item             | Cost|
 * |:---------------------------|:---:|
 * |Test odds with likely       | 18s |
 * |Test odds with unlikely     | 21s |
 * |Test odds without likely    | 20s |
 * |Test evens without unlikely | 22s |
 */


