#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm> 
#include <ctime>


using namespace std;

typedef struct
{
	char word[20];
	int count;
	int length;
} word_struct;


// GPU method
/* keyLen - Length of work to be processed by each thread
* dataLen - Total size of data
* data - Pointer to the start of the data
* keyword - Pointer to the start of the keyword
*/
__global__ void searchText(char* data, word_struct* outputArray, int dataLen, int numWords)
{
	int threadIndex = blockIdx.x *blockDim.x + threadIdx.x;
	int instances = 0;
	word_struct thisWord;
	thisWord.length = 0;
	thisWord.count = 0;
	// Initialise the Char array to only spaces
	for (int i = 0; i < 20; i++)
	{
		thisWord.word[i] = ' ';
	}

	/* ----------------------- Find Word by Thread Index ------------------------------*/

		int startOfWord = 0; // Holds the index of the start of a word after the space ->' 't'e's't'
		int lengthOfWord = 0; //Holds the length of the word before the next space
		int spacesCount = 0; // Number of spaces, used to calculate if this threads word has been found
		int wordIndex = 0; // Used to walk the keyword as we walk the data array to check for matches
		bool wordFound = false; // Used to identify if the word was found
		// Find the Keyword I'm looking for
		for (int j = 0; j < dataLen - 1; j++)
		{
			if (data[j] == ' ' && data[j + 1] != ' ') {
				spacesCount++;
				// Start of the word has been found
				if (spacesCount == threadIndex) {
					startOfWord = j + 1;
				}
				// End of the word has been found
				if (spacesCount == threadIndex + 1) {
					lengthOfWord = j - startOfWord;
					wordIndex = 0;
					wordFound = true;
					for (int i = startOfWord; i < startOfWord + lengthOfWord; i++) {
						if (i < startOfWord + 19) {
							thisWord.word[wordIndex] = data[i];
							wordIndex++;
						}
					}
					thisWord.length = lengthOfWord;
					break; // Performance Enhancement
				}
			}
		}

		/* ------------------------------------------------------------------------------------*/

		/* ------------------ Find instances of the word in the data set ----------------------*/

		if (wordFound) {
			int keywordIndex = startOfWord;

			for (int j = 0; j < dataLen - 1; j++)
			{
				if (data[j] == data[keywordIndex])
				{
					keywordIndex++;
					if (keywordIndex == startOfWord + lengthOfWord)
					{ // A full word has been found -
						instances++;
						// Start the keyword from origin again
						keywordIndex = startOfWord;
					}

				}
				else { // The current word doesn't match our keyword
					// Start the keyword from origin again
					keywordIndex = startOfWord;
				}

			}

			/* ------------------------------------------------------------------------------------*/

			/* ------------------ Output data to the console ----------------------*/

			thisWord.count = instances;
			if (spacesCount <= numWords) {
				outputArray[spacesCount - 1] = thisWord;
			}


			/* --------------------------------------------------------------------*/
		}
}

/* ------------------- CUDA Error Handler -----------------------/
Handles Error output for all CUDA operations
/---------------------------------------------------------------*/

void _checkCudaError(char* message, cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		system("pause");
		exit(0);
	}
}

/* -------------------------------------------------------------*/

bool word_sorter(word_struct const& lhs, word_struct const& rhs) {
	// Returns the highest count
	return lhs.count > rhs.count;
}

bool compare_words(word_struct const& lhs, word_struct const& rhs) {
	// Returns -1 if words are not even
	return std::strcmp(lhs.word, rhs.word) < 0;
}

bool remove_words(word_struct const& lhs, word_struct const& rhs) {
	// Returns true if the words are exact matches
	return std::strcmp(lhs.word, rhs.word) == 0;
}

int main(int argc, char* argv[])
{

	/* ------------------- Initialisations ---------------------------/
	Initialise variables for the device and host
	/---------------------------------------------------------------*/

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Dynamic allocation of threads based on GPU hardware
	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, 0);
	int numThreads = deviceProperties.maxThreadsPerBlock;

	char* text_input = (char*)malloc(512 * sizeof(char));

	/* ------------------- Get User Input ---------------------------/
	Read the name of the files for the string input and stopword dictionary
	/---------------------------------------------------------------*/
	printf("Available Files:\n\n");
	printf("Sample.txt : 478\n");
	printf("Macbook_3k_chars.txt : 3008\n");
	printf("Sheffield_Hallam.txt : 15059\n");
	printf("Cappucino.txt : 35307\n");
	printf("CUDA_50k_chars.txt : 51615\n");
	printf("GoLang_60k_chars.txt : 60438\n");
	printf("Logitech_100k_chars.txt : 83360\n");
	printf("NVIDIA_150k_chars.txt : 143580\n\n");

	printf("Enter the input file name which has to be searched\n");
	scanf("%s", text_input);
	printf("input = %s", text_input);

	/* ----------------------- Read Files ---------------------------/
	Read the input string and the stopwords dictionary from a file.
	/---------------------------------------------------------------*/
	FILE *f = fopen(text_input, "r");
	// Find the end of the file
	fseek(f, 0, SEEK_END);
	// Save the file size
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	// Devices pointer to memory the size of the file
	char *text = (char *)malloc((fsize + 1) * sizeof(char));
	// Read the file into that memory
	fread(text, fsize, 1, f);

	/* ----------------------- Convert String to Vector ---------------------------/
	To allow us to count the words and appropriately allocate the struct space.
	/-----------------------------------------------------------------------------*/

	std::clock_t cpuStart;
	double cpuPreProcessDuration, cpuPostProcessDuration;

	cpuStart = std::clock();

	string stringText;
	stringText.assign(text, fsize);
	// Make lower case
	transform(stringText.begin(), stringText.end(), stringText.begin(), ::tolower);
	// std::replace_if(stringText.begin(), stringText.end(), ::isdigit, ' ');
	// std::replace_if(stringText.begin(), stringText.end(), ::ispunct, ' ');

	// create a stringstream for our text file
	stringstream ss(stringText);
	// Create two vector iterators
	istream_iterator<string> begin(ss);
	istream_iterator<string> end;
	vector<string> vstrings(begin, end);
	// Copy our data from our string into the vector
	std::copy(vstrings.begin(), vstrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

	cpuPreProcessDuration = (std::clock() - cpuStart) / (double)CLOCKS_PER_SEC;

	int numElements = vstrings.size();

	printf("\nNumber of Words to process:  %d", numElements);

	int sizeOfWordStructArray = (numElements * sizeof(word_struct));

	printf("\nBytes required to store device response:  %d", sizeOfWordStructArray);

	/* ----------------------------------------------------------------------------- */

	/* ----------------------- Pre Kernel Tasks -----------------------------------/
	Assign and allocate Memory, Blocks and Threads.
	/-----------------------------------------------------------------------------*/

	// Output array on Device
	word_struct* d_wordArray = (word_struct*)malloc(sizeOfWordStructArray);
	// Output array on Host
	word_struct* h_wordArray = (word_struct*)malloc(sizeOfWordStructArray);

	cudaMalloc((void**)&d_wordArray, sizeOfWordStructArray);

	printf("\nFile reading complete...");
	fclose(f);

	int noOfBlocks = strlen(text) / numThreads;
	noOfBlocks++;
	printf("\nBlock size = %d\nFilesize = %d\n", noOfBlocks, fsize);

	char* d_text; // Pointer to the text on the device
	// Allocate memory based on length of string * the memory capacity of a char
	cudaMalloc((void**)&d_text, strlen(text) * sizeof(char));
	// Copy text into device variable d_text
	cudaMemcpy(d_text, text, strlen(text) * sizeof(char), cudaMemcpyHostToDevice);

	/* ---------------------------------------------------------------------------- - */

	/* ----------------------------- Kernel Call -----------------------------/
	Call Kernel and report errors sensibly.
	/------------------------------------------------------------------------*/

	// Error handler, prints pre-defined messages to help debugging
	_checkCudaError(
		"Memory Copy To Device",
		cudaGetLastError()
		);

	cpuStart = std::clock();
	cudaEventRecord(start);

	/* Calls searchText Kernel, with:
	* d_text : pointer to the text input
	* d_wordArray : array to store the number of word structs
	* datalength : How big our dataset is
	*/
	printf("Sending %d elements to the Kernel", numElements);

	searchText << <noOfBlocks, numThreads >> >(d_text, d_wordArray, strlen(text), numElements);


	cudaGetLastError();
	// Error handler, prints pre-defined messages to help debugging
	_checkCudaError(
		"kernel launch",
		cudaGetLastError()
		);

	cudaDeviceSynchronize();
	// Error handler, prints pre-defined messages to help debugging
	_checkCudaError(
		"Synchronisation",
		cudaGetLastError()
		);
	// Copy the contents of the device array to our host array
	cudaMemcpy(h_wordArray, d_wordArray, sizeOfWordStructArray, cudaMemcpyDeviceToHost);
	// Error handler, prints pre-defined messages to help debugging
	_checkCudaError(
		"Memory Copy From Device",
		cudaGetLastError()
		);

	cudaEventRecord(stop);
	float milliseconds = (std::clock() - cpuStart) / (double)CLOCKS_PER_SEC;


	/* --------------------------------------------------------------------- */

	/* ----------------------- Post Process Data ----------------/
						Prepare the data for output
	/-----------------------------------------------------------*/

	printf("\nNumber of Results: %d", numElements);
	// Time this process
	cpuStart = std::clock();

	// Create a vector to hold all our unique elements
	std::vector<word_struct> uniqueElements;
	// Assign the contents of h_wordArray
	uniqueElements.assign(h_wordArray, h_wordArray + numElements);
	// Sort the elements using the compare_words function
	std::sort(uniqueElements.begin(), uniqueElements.end(), &compare_words);
	// Create an iterator which uses the remove_words function
	vector<word_struct>::iterator newEnd = unique(uniqueElements.begin(), uniqueElements.end(), &remove_words);
	// Call the iterator on the vector to remove non-unique values
	uniqueElements.erase(newEnd, uniqueElements.end());
	// Sort the vector back into count descending order
	std::sort(uniqueElements.begin(), uniqueElements.end(), &word_sorter);

	/* --------------------------------------------------------------------- */


	/* ----------------------- Print Data ----------------------/
					Output the data meaningfully.
	/-----------------------------------------------------------*/

	cudaEventElapsedTime(&milliseconds, start, stop);
	cpuPostProcessDuration = (std::clock() - cpuStart) / (double)CLOCKS_PER_SEC;

	printf("\n|------------------------------------------------------|");
	printf("\n|-------------------- Completed! ----------------------|");
	printf("\n|------------------------------------------------------|");
	printf("\n| File Searched: %s                              ", text_input);
	printf("\n|------------------------------------------------------|");
	printf("\n| Number of Blocks Used: %d                              ", noOfBlocks);
	printf("\n| Number of Threads Used: %d                              ", numThreads);
	printf("\n| Words to process: %d                              ", numElements);
	printf("\n|------------------- Time Taken -----------------------|");
	printf("\n| Time Taken to preprocess data: %fms                              ", cpuPreProcessDuration);
	printf("\n| Time Taken to process data on GPU: %fms                            ", milliseconds);
	printf("\n| Time Taken to postprocess data on CPU: %fms                              ", cpuPostProcessDuration);
	printf("\n| Total Time Taken: %fms                              ", cpuPreProcessDuration + milliseconds + cpuPostProcessDuration);
	printf("\n|------------------------------------------------------|");
	printf("\n|-------------------- Top Ten Results! ----------------|");
	printf("\n|------------------------------------------------------|");

	// Number of printed results
	int printed = 0;

	for (int i = 0; printed < 10; ++i)
	{
		// If the words are longer than two characters they are probably not stop words
		if (uniqueElements[i].length > 2) {
			printf("\n|");
			// Print the whole c-string
			for (int j = 0; j < 19; j++) {
				printf("%c", uniqueElements[i].word[j]);
			}
			printf(" : %d", uniqueElements[i].count);
			++printed;
		}
		else {

		}
	}
	printf("\n|------------------------------------------------------|\n\n\n");

	/* -------------------------------------------------------- */

	cudaFree(d_text);
	free(text);

	system("pause");
	return 0;
}