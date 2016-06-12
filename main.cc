/* MarchSAT, Copyright (c) 2003 Jost Neigenfind <jostie@gmx.de>
 * WalkSAT for GPUs
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#include <CL/cl.h>

#include <boost/config.hpp>
#include <boost/program_options/detail/config_file.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct my_pair{
	uint32_t first;
	uint32_t second;
};

bool pairSorterSecondDown(my_pair const& pair1, my_pair const& pair2){
	return pair1.second > pair2.second;
}

bool pairSorterSecondUp(my_pair const& pair1, my_pair const& pair2){
	return pair1.second < pair2.second;
}

struct program_options {
    std::string infile;
    std::string intermediates;
    std::string solution;
    std::string binary_dump;
    std::string binary_intermediates;
    uint32_t nr_threads;
    uint32_t nr_workgroups;
    uint32_t nr_iterations;
    double random_flips;
    double reuse;
    int show_intermediate;
    int device;
    bool crossover;
    bool dump;
    bool dump2;
    bool compare;
};

struct thread_state {
	uint64_t rnd;
	uint32_t nr_satisfied;
};

struct gpu_clause {
	uint32_t literals[4];
};

struct object_info {
	uint32_t size;
	uint32_t* address;
};

void CL_CALLBACK pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "notify: %s\n", errinfo);
}

typedef int literal;
typedef unsigned int variable;
typedef std::map<variable, variable> variable_map;
typedef std::vector<literal> clause;
typedef std::vector<clause> clause_vector;

// ###########################################################################################
// #            methods that handle and evaluate variables, literals and clauses             #
// ###########################################################################################

uint8_t get_bit(uint32_t *bit_values, uint32_t var, uint32_t nr_bit_words, uint32_t thread_id)
{
	uint32_t index = var >> 5;
	uint32_t offset = var % 32;

	uint32_t word = bit_values[index + nr_bit_words*thread_id];
	return (word << (32 - offset - 1)) >> 31;
}

bool evaluateClause(clause_vector &clauses, uint32_t clause_index, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t thread_id){
    clause &c = clauses[clause_index];

    bool value = false;
    for (unsigned int k = 0; k < c.size(); ++k) {
        literal l = c[k];
        unsigned int var = abs(l);
        bool sign = (l < 0);

        value |= get_bit(bit_values, var, nr_bit_words, thread_id) ^ sign;
    }

    return value;
}

// --------------------------------------------------------------------------------------

bool get_bit_(uint32_t var, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t thread_id)
{
	uint32_t index = var >> 5;
    uint32_t offset = var % 32;

	uint32_t word = bit_values[index + nr_bit_words*thread_id];

	return (word << (31 - offset)) >> 31;
}

void flip_bit_(uint32_t var, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t thread_id)
{
	uint32_t index = var >> 5;
    uint32_t offset = var % 32;

	uint32_t word = bit_values[index + nr_bit_words*thread_id];
	word ^= 1 << offset;

	bit_values[index + nr_bit_words*thread_id] = word;
}

bool evaluate_literal_(uint32_t lit, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t flip_var, uint32_t thread_id)
{
	bool sign = lit & 1;
	uint32_t var = lit >> 1;

	uint32_t index = var >> 5;
    uint32_t offset = var % 32;

	uint32_t word = bit_values[index + nr_bit_words*thread_id];
	uint32_t evaluation = (word << (31 - offset)) >> 31;

	bool flip = var == flip_var;
	return (evaluation ^ sign) ^ flip;
}

bool evaluate_clause_(uint32_t* clauses, uint32_t index_to_clause, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t flip_var, uint32_t thread_id)
{
	// this function behaved weird and therefore this strange return pattern
	// there was a bug in one of the calling functions, might have been due to that

	uint32_t index 	= clauses[index_to_clause];		// get pointer to clause info
	uint32_t nr_lits = clauses[index];				// first field contains number of literals

	for (int i = 0; i < nr_lits; i++){
		uint32_t lit = clauses[index + 1 + i];
		if(evaluate_literal_(lit, bit_values, nr_bit_words, flip_var, thread_id))
			return true;
	}
	return false;
}

uint32_t evaluate_clauses_of_variable_unsatisfied_(uint32_t var, uint32_t *variables_to_clauses, uint32_t *clauses, uint32_t *bit_values, uint32_t nr_bit_words, bool flip, uint32_t thread_id){
	uint32_t index = variables_to_clauses[var]; 			// get index where variable to clauses info is stored
	uint32_t nr_clauses = variables_to_clauses[index];	// first field contains number of clauses the variable occurs in

	uint32_t evaluation = 0;
	for (unsigned int i = 0; i < nr_clauses; i++){
		uint32_t index_to_clause = variables_to_clauses[index + 1 + i];
		if (flip)
			evaluation += !evaluate_clause_(clauses, index_to_clause, bit_values, nr_bit_words, var, thread_id);
		else
			evaluation += !evaluate_clause_(clauses, index_to_clause, bit_values, nr_bit_words, 0, thread_id);
	}

	return evaluation;
}

void evaluate_clauses_of_variable_unsatisfied2_(uint32_t var, uint32_t *variables_to_clauses, uint32_t *clauses, uint32_t *bit_values, uint32_t nr_bit_words, bool flip, uint32_t thread_id, std::map<unsigned int, unsigned int> &clause_collection){
	uint32_t index = variables_to_clauses[var]; 			// get index where variable to clauses info is stored
	uint32_t nr_clauses = variables_to_clauses[index];	// first field contains number of clauses the variable occurs in

	for (unsigned int i = 0; i < nr_clauses; i++){
		uint32_t index_to_clause = variables_to_clauses[index + 1 + i];
		if (flip){
			if (!evaluate_clause_(clauses, index_to_clause, bit_values, nr_bit_words, var, thread_id))
                clause_collection[index_to_clause]++;
        } else{
			if (!evaluate_clause_(clauses, index_to_clause, bit_values, nr_bit_words, 0, thread_id))
                clause_collection[index_to_clause]++;
        }
	}
}

// ###########################################################################################

void makeDeviceVariablesToClauses(clause_vector &clauses, unsigned int nr_variables, struct object_info &info){
	std::vector<std::vector<int> > vars_to_clauses;
	for (unsigned int i = 0; i < nr_variables; i++){
		std::vector<int> v;
		vars_to_clauses.push_back(v);
	}

	for (unsigned int i = 0; i < clauses.size(); i++){
		clause &c = clauses[i];
		for (unsigned int j = 0; j < c.size(); j++){
			literal l = c[j];
			unsigned int var = abs(l);

			std::vector<int> &v = vars_to_clauses[var];
			v.push_back(i);
			vars_to_clauses[var] = v;
		}
	}

	// for debugging
	/*for (unsigned int i = 0; i < vars_to_clauses.size(); i++){
		std::vector<int> &v = vars_to_clauses[i];
		printf(" %i", i);
		for (unsigned int j = 0; j < v.size(); j++){
			printf(" %i", v[j]);
		}
		printf("\n");
	}*/

	int body = 0;
	for (unsigned int i = 0; i < vars_to_clauses.size(); i++){
		body++;
		std::vector<int> &v = vars_to_clauses[i];
		body = body + v.size();
	}

	info.size = nr_variables + body;
	info.address = new uint32_t[info.size];
	uint32_t* device_variables_to_clauses = info.address;

	body = 0;
	for (unsigned int i = 0; i < vars_to_clauses.size(); i++){
		std::vector<int> &v = vars_to_clauses[i];	// get clause mapping for variable i

		device_variables_to_clauses[i] = nr_variables + body;	// save index where list of clause indices is stored
		device_variables_to_clauses[nr_variables + body] = v.size();	// save number of clauses
		body++;

		// save list of clause indices
		for (unsigned int j = 0; j < v.size(); j++){
			device_variables_to_clauses[nr_variables + body] = v[j];
			body++;
		}
	}
}

void makeDeviceClauses(clause_vector &clauses, struct object_info &info){
	int body = 0;	// clause size counter
	for (unsigned int i = 0; i < clauses.size(); i++){
		body++;	// make space for the number of literals
		clause &c = clauses[i];
		body = body + c.size();	// make space for the literals themselfs
	}

	// create the clause buffer which consists of nr_clauses pointers and space
	// for the non trivial clauses and the trivial clauses
	info.size = clauses.size() + body;
	info.address = new uint32_t[info.size];
	uint32_t* device_clauses = info.address;

	body = 0;	// use this as counter for the current clause and literal
	for (unsigned int i = 0; i < clauses.size(); i++){
		clause &c = clauses[i];	// get ith clause

		device_clauses[i] = clauses.size() + body;	// save index of clause
		device_clauses[clauses.size() + body] = c.size();	// save size of clause
		body++;	// increase counter for clauses and literals

		// loop over literals
		for (unsigned int j = 0; j < c.size(); j++){
			literal l = c[j];				// get literal
			unsigned int var = abs(l);		// get variable which is equivalent to index of index
			unsigned int sign = (l < 0);	// get sign

			assert(var < ((1U << 31) - 1));	// test if index uses at most 31 bits
			// because we need the first bit for the sign

			// save literal at corresponding position
			device_clauses[clauses.size() + body] = (var << 1) | sign;
			body++;
		}
	}
}

void readCnf(const char *filename, variable_map &variables, variable_map &reverse_variables, clause_vector &clauses)
{
	bool redundant_literal = false;
	std::ifstream file;

	file.open(filename);
	if (!file) {
		fprintf(stderr, "ifstream::open() failed\n");
		exit(EXIT_FAILURE);
	}

	int line_counter = 0;
	while (!file.eof()) {
		line_counter++;

		std::string line;
		getline(file, line);

		if (line.size() == 0)
			continue;

		/* Skip problem line -- we don't use it anyway */
		if (line[0] == 'p')
			continue;

		/* Skip comments */
		if (line[0] == 'c')
			continue;

		/* XOR clauses */
		if (line[0] == 'x') {
			fprintf(stderr, "Cannot read XOR clauses\n");
			exit(EXIT_FAILURE);
		}

		clause c;
		std::stringstream s(line);
		std::map<int, unsigned int> same_literal;
		while (!s.eof()) {
			literal l;
			s >> l;

			if (l == 0)
				break;

			if (same_literal.count(l) == 0){
				variable v = abs(l);

				// remap of variables to the range [1, n], where n is the total number of variables
				variable v2;
				variable_map::iterator it = variables.find(v);
				if (it == variables.end()) {
					v2 = 1 + variables.size();
					variables[v] = v2;
					reverse_variables[v2] = v;
				} else {
					v2 = it->second;
				}

				c.push_back(v2 * (l < 0 ? -1 : 1));
			} else {
                if (!redundant_literal){
                	printf("v parsing information ...\n");
                	redundant_literal = true;
                }
				printf("v    literal %i occuring multiple times in clause in line %i\n", l, line_counter);
			}
			same_literal[l]++;
		}
		clauses.push_back(c);
	}

	file.close();
	if (redundant_literal)
        printf("v ... done\n");
}

unsigned int	nr_threads;
unsigned int 	nr_variables;
unsigned int	nr_bit_words;
variable_map	reverse_variables;
variable_map    variables;
clause_vector	clauses;
uint32_t		*host_bit_values;
thread_state	*host_threads;
program_options prog_opts;

void readIntermediatesBinary(const char *filename, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t nr_threads){
    FILE* file = fopen(filename , "rb" );

    fseek(file, 0, SEEK_END);
    unsigned int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned int buffer_size = nr_threads*nr_bit_words*sizeof(uint32_t);
    if (file_size < buffer_size){
        unsigned int times = buffer_size/file_size;
        unsigned int rest = buffer_size%file_size;

        unsigned int offset = 0;
        for (unsigned int i = 0; i < times; i++){
            fread(&bit_values[offset], 1, file_size, file);
            fseek(file, 0, SEEK_SET);
            offset = offset + (file_size/sizeof(uint32_t)); //file_size is 1 byte and bit_values is 4 byte
        }
        if (rest > 0)
            fread(&bit_values[offset], 1, rest, file);
    } else {
        fread(bit_values, 1, buffer_size, file);
    }

    fclose (file);
}

void readIntermediates(const char *filename, variable_map &variables, uint32_t *bit_values, uint32_t nr_bit_words, uint32_t nr_threads){
	// set everything to zero
	for (unsigned int i = 0; i < nr_threads; i++){
		for (unsigned int j = 0; j < nr_bit_words; j++){
			bit_values[i*nr_bit_words + j] = 0;
		}
	}

	std::ifstream file;

	file.open(filename);
	if (!file) {
		fprintf(stderr, "ifstream::open() failed\n");
		exit(EXIT_FAILURE);
	}

	unsigned int line_counter = 0;
	while (!file.eof()) {
		std::string line;
		getline(file, line);

		if (line.size() == 0)
			continue;

		/* Skip problem line -- we don't use it anyway */
		if (line[0] == 'p')
			continue;

		/* Skip comments */
		if (line[0] == 'c')
			continue;

		/* Skip verbose */
		if (line[0] == 'v')
			continue;

		/* XOR clauses */
		if (line[0] == 'x') {
			fprintf(stderr, "Cannot read XOR clauses\n");
			exit(EXIT_FAILURE);
		}

        //printf("line counter = %i\n", line_counter);
		std::stringstream s(line);
		while (!s.eof()) {
			literal l;
			s >> l;

			if (l == 0)
				break;

			bool sig = (l < 0);
			variable v = abs(l);
			variable v2 = variables[v];

			uint32_t index = v2 >> 5;
			uint32_t offset = v2 % 32;

			uint32_t word = bit_values[line_counter*nr_bit_words + index];
			bit_values[line_counter*nr_bit_words + index] = word | ((!sig) << offset);
		}
		line_counter++;
		if (line_counter >= prog_opts.nr_threads)
            break;
	}

	// if there are more thread "slots" than saved vectors
	if (line_counter < prog_opts.nr_threads){
        int c = 0;
        while (line_counter + c < prog_opts.nr_threads){
            int index = (line_counter + c)%line_counter;

            for (unsigned int i = 0; i < nr_bit_words; i++)
                bit_values[(line_counter + c)*nr_bit_words + i] = bit_values[index + i];

            c++;
        }
	}

	file.close();
}

void readSolution(const char *filename, variable_map &variables, uint32_t *bit_values, uint32_t nr_bit_words){
	// set everything to zero
    for (unsigned int j = 0; j < nr_bit_words; j++)
        bit_values[j] = 0;

	std::ifstream file;

	file.open(filename);
	if (!file) {
		fprintf(stderr, "ifstream::open() failed\n");
		exit(EXIT_FAILURE);
	}

	while (!file.eof()) {
		std::string line;
		getline(file, line);

		if (line.size() == 0)
			continue;

		/* Skip problem line -- we don't use it anyway */
		if (line[0] == 'p')
			continue;

		/* Skip comments */
		if (line[0] == 'c')
			continue;

		/* Skip verbose */
		if (line[0] == 'v')
			continue;

		/* XOR clauses */
		if (line[0] == 'x') {
			fprintf(stderr, "Cannot read XOR clauses\n");
			exit(EXIT_FAILURE);
		}

		std::stringstream s(line);
		while (!s.eof()) {
			literal l;
			s >> l;

			if (l == 0)
				break;

			bool sig = (l < 0);
			variable v = abs(l);
			variable v2 = variables[v];

			uint32_t index = v2 >> 5;
			uint32_t offset = v2 % 32;

			uint32_t word = bit_values[index];
			bit_values[index] = word | ((!sig) << offset);
		}
	}

	file.close();
}

void getProgramOptions(int argc, char *argv[], struct program_options &prog_opts){
    // default values
    prog_opts.infile = "";
    prog_opts.intermediates = "";
    prog_opts.solution = "";
    prog_opts.binary_dump = "";
    prog_opts.binary_intermediates = "";
    prog_opts.nr_threads = 256;
    prog_opts.nr_workgroups = 1;
    prog_opts.nr_iterations = prog_opts.nr_threads*256;
    prog_opts.random_flips = 0.25;
    prog_opts.reuse = 1.0;
    prog_opts.device = 0;
    prog_opts.show_intermediate = -1;
    prog_opts.crossover = false;
    prog_opts.dump = false;
    prog_opts.dump2 = false;
    prog_opts.compare = false;

	po::options_description description("runsat: ./runsat --infile <file> [options]");
	description.add_options()
		("help", "displays this help message")
		("version", "displays version number")
		("crossover", "uses crossing over")
		("dump", "dumps intermediates if interrupted")
		("dump2", "also dumps adpated clauses")
		("compare", "compare clauses")
		("reuse", po::value<double>(), "sets fraction of reused solutions")
		("infile", po::value<std::string>(), "reads input file")
		("intermediates", po::value<std::string>(), "reads dump from previous run")
		("solution", po::value<std::string>(), "reads solution from file and tests it")
		("binary-dump", po::value<std::string>(), "dumps intermediates to binary file if interrupted")
		("binary-intermediates", po::value<std::string>(), "reads binary dump from file")
		("iterations", po::value<int>(), "sets number of iterations")
		("threads", po::value<int>(), "sets number of threads")
		("workgroups", po::value<int>(), "sets number of workgroups")
		("flips", po::value<double>(), "sets frequency of random flips")
        ("device", po::value<int>(), "sets device to use")
        ("show-intermediate", po::value<int>(), "show intermediate with given number");

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
		po::notify(vm);
	} catch(const po::error &e){
		std::cerr << e.what() << std::endl;
		exit(0);
	}

	if (vm.count("help") || vm.size() == 0){
		std::cout << description;
		exit(0);
	}

	if (vm.count("version")){
		std::cout << "v0.01\n";
		exit(0);
	}

	if (vm.count("crossover"))
		prog_opts.crossover = true;

	if (vm.count("dump"))
		prog_opts.dump = true;

	if (vm.count("dump2"))
		prog_opts.dump2 = true;

    if (vm.count("compare"))
        prog_opts.compare = true;

	if (vm.count("reuse"))
		prog_opts.reuse = vm["reuse"].as<double>();

	if (vm.count("infile"))
		prog_opts.infile = vm["infile"].as<std::string>();

	if (vm.count("intermediates"))
		prog_opts.intermediates = vm["intermediates"].as<std::string>();

	if (vm.count("solution"))
		prog_opts.solution = vm["solution"].as<std::string>();

	if (vm.count("binary-dump"))
		prog_opts.binary_dump = vm["binary-dump"].as<std::string>();

	if (vm.count("binary-intermediates"))
		prog_opts.binary_intermediates = vm["binary-intermediates"].as<std::string>();

	if (vm.count("threads"))
		prog_opts.nr_threads = vm["threads"].as<int>();

	if (vm.count("workgroups"))
		prog_opts.nr_workgroups = vm["workgroups"].as<int>();

	if (vm.count("iterations"))
		prog_opts.nr_iterations = vm["iterations"].as<int>();

	if (vm.count("flips"))
		prog_opts.random_flips = vm["flips"].as<double>();

    if (vm.count("device"))
        prog_opts.device = vm["device"].as<int>();

    if (vm.count("show-intermediate"))
        prog_opts.show_intermediate = vm["show-intermediate"].as<int>();


    if (prog_opts.infile.size() == 0)
		std::cout << description;

	printf("v listing parameter ...\n");
	printf("v    input file:\t\t\t%s\n", prog_opts.infile.c_str());
	printf("v    intermediates file:\t\t%s\n", prog_opts.intermediates.c_str());
	printf("v    solutions file:\t\t\t%s\n", prog_opts.solution.c_str());
	printf("v    binary-dump file:\t\t\t%s\n", prog_opts.binary_dump.c_str());
	printf("v    binary-intermediates file:\t\t%s\n", prog_opts.binary_intermediates.c_str());
	printf("v    number of threads:\t\t\t%i\n", prog_opts.nr_threads);
	printf("v    number of workgroups:\t\t%i\n", prog_opts.nr_workgroups);
	printf("v    number of iterations:\t\t%i\n", prog_opts.nr_iterations);
	printf("v    frequency of random flips:\t\t%f\n", prog_opts.random_flips);
    printf("v    fraction of reused data:\t\t%f\n", prog_opts.reuse);
	printf("v    crossing over:\t\t\t%i\n", prog_opts.crossover);
    printf("v    device used:\t\t\t%i\n", prog_opts.device);
    printf("v    show intermediate:\t\t\t%i\n", prog_opts.show_intermediate);
	printf("v    dump intermdiates:\t\t\t%i\n", prog_opts.dump);
	printf("v    also dump adapted clauses:\t\t%i\n", prog_opts.dump2);
	printf("v    compare clauses:\t\t\t%i\n", prog_opts.compare);
	printf("v ... done\n");
}

void getDeviceInfo(){
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

	printf("v listing devices ... \n");
    for (unsigned i = 0; i < platformCount; i++) {
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (unsigned j = 0; j < deviceCount; j++) {
            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("v    device %i: %s\n", j, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf("v       hardware version:\t\t%s\n", value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf("v       software version:\t\t%s\n", value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf("v       OpenCL C version:\t\t%s\n", value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf("v       parallel compute units:\t\t%d\n", maxComputeUnits);

			size_t max_work_group_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
            printf("v       maximal work group size:\t%zu\n", max_work_group_size);

			cl_ulong max_mem_alloc_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
            printf("v       maximal memory allocation size:\t%lu\n", max_mem_alloc_size);
        }
        free(devices);

    }
    free(platforms);
	printf("v ... done\n");
}

void printSolution(variable_map &reverse_variables, unsigned int nr_variables, clause_vector &clauses, uint32_t *bit_values, unsigned int nr_bit_words, unsigned int thread_id, bool test){
    // if test is true, then check current assigned if being satisfiable
    if (test){
        // Double-check on the host that this is a correct solution.
        for (unsigned int j = 0; j < clauses.size(); ++j) {
            bool value = evaluateClause(clauses, j, bit_values, nr_bit_words, thread_id);
            if (!value){
                printf(" %i. ", j);

                clause &c = clauses[j];
                for (unsigned int k = 0; k < c.size(); ++k) {
                    literal l = c[k];
                    unsigned int var = abs(l);
                    bool sign = (l < 0);

                    variable v = reverse_variables[var];
                    if (sign)
                        printf(" -%i", v);
                    else
                        printf(" %i", v);
                }
                printf(" 0\n");
            }
			assert(value);
        }
	}

    // get original variable name and its truth assignment, put it into a pair structure
	my_pair pairs[nr_variables - 1];
	for (unsigned int j = 1; j < nr_variables; ++j) {
		unsigned int v = reverse_variables[j];

		pairs[j - 1].first = get_bit(bit_values, j, nr_bit_words, thread_id) ? v : -v;
		pairs[j - 1].second = v;
	}

    // sort it by original variable name (would also work with different sorter and absolute value)
	std::sort(pairs, pairs + nr_variables - 1, &pairSorterSecondUp);

    // print original name with assignment
	for (unsigned int j = 1; j < nr_variables; ++j)
		printf(" %d", pairs[j - 1].first);
	printf(" 0\n");
}

void compareClauses(variable_map &reverse_variables, unsigned int nr_variables, clause_vector &clauses, uint32_t *bit_values, unsigned int nr_bit_words, unsigned int nr_threads){
    bool* solved = new bool[nr_threads*clauses.size()];
    for (unsigned int i = 0; i < nr_threads; i++){
        printf("reading clauses of %i\n", i);
        variable_map unsat_vars;

        // get variables in unsatisfied clauses
        for (unsigned int j = 0; j < clauses.size(); j++) {
            bool value = evaluateClause(clauses, j, bit_values, nr_bit_words, i);
            if (!value){
                clause &c = clauses[j];
                for (unsigned int k = 0; k < c.size(); k++){
                    literal l = c[k];
                    unsigned int var = abs(l);
                    unsat_vars[var] = 1;
                }
            }
        }

        for (unsigned int j = 0; j < clauses.size(); j++){
            clause &c = clauses[j];

            bool value = false; // variable in unsatisfied clause
            for (unsigned int k = 0; k < c.size(); k++){
                literal l = c[k];
                unsigned int var = abs(l);

                if (unsat_vars.count(var) == 1)
                    value = true;
            }
            solved[i*clauses.size() + j] = !value;  // which clauses consist of variables only occuring in satsified clauses
        }
    }

    int min_c = -1;
    for (unsigned int i = 0; i < nr_threads; i++){
        for (unsigned int j = i + 1;  j < nr_threads; j++){
            printf("comparing %i with %i ... ", i, j);
            unsigned int c = 0;
            for (unsigned int k = 0; k < clauses.size(); k++){
                if (!solved[i*nr_threads + k] && !solved[j*nr_threads + k])
                    c++;
            }
            printf("overlapp of %i\n", c);
            if (min_c == -1 || c < min_c)
                min_c = c;
        }
    }
    printf("minimal overlapp: %i\n", min_c);

    delete[] solved;
}

unsigned int printSimplified(variable_map &reverse_variables, unsigned int nr_variables, clause_vector &clauses, uint32_t *bit_values, unsigned int nr_bit_words, unsigned int thread_id, bool print){
    // irgendwas stimmt hier noch nicht
    variable_map unsat_vars;

    // get variables in unsatisfied clauses
	for (unsigned int j = 0; j < clauses.size(); j++) {
        bool value = evaluateClause(clauses, j, bit_values, nr_bit_words, thread_id);
		if (!value){
            clause &c = clauses[j];
            for (unsigned int k = 0; k < c.size(); k++){
                literal l = c[k];
                unsigned int var = abs(l);
                unsat_vars[var] = 1;
            }
		}
	}

    bool* solved = new bool[clauses.size()];
	for (unsigned int j = 0; j < clauses.size(); j++){
        clause &c = clauses[j];

        bool value = false; // variable in unsatisfied clause
        for (unsigned int k = 0; k < c.size(); k++){
            literal l = c[k];
            unsigned int var = abs(l);

            if (unsat_vars.count(var) == 1)
                value = true;
        }
        solved[j] = !value;  // which clauses consist of variables only occuring in satsified clauses
	}

    unsigned int counter = 0;
	for (unsigned int j = 0; j < clauses.size(); j++){
        if (!solved[j]){
            if (print){
                clause &c = clauses[j];

                printf("s%i ", thread_id);
                for (unsigned int k = 0; k < c.size(); k++){
                    literal l = c[k];
                    unsigned int var = abs(l);
                    //unsat_vars[var] = 1;    // additional variables which are not set yet <= das ist zmdst nicht korrekt

                    bool sign = (l < 0);

                    variable v = reverse_variables[var];
                    if (sign)
                        printf(" -%i", v);
                    else
                        printf(" %i", v);
                }
                printf(" 0\n");
            }
            counter++;
        }
	}

    delete[] solved;

    if (print){
    for (unsigned int var = 1; var < nr_variables; var++){
        if (unsat_vars.count(var) == 0){
            bool b = get_bit(bit_values, var, nr_bit_words, thread_id);
            variable v = reverse_variables[var];
            if (b)
                printf("s%i %i 0\n", thread_id, v);
            else
                printf("s%i -%i 0\n", thread_id, v);
        }
    }
    }

	return counter;
}

void printCurrentAll(){
	for (unsigned int i = 0; i < prog_opts.nr_threads; i++){
        if (prog_opts.dump2)
            printSimplified(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, i, true);
        printSolution(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, i, false);
    }
}

void saveCurrentAllBinary(){
    FILE* file;
    file = fopen(prog_opts.binary_dump.c_str(), "wb");
    fwrite(host_bit_values, 1, prog_opts.nr_threads*nr_bit_words*sizeof(uint32_t), file);
    fclose(file);
}

void my_handler(int s){
	printf("caught signal %d\n", s);
	if (prog_opts.dump)
		printCurrentAll();
    if (prog_opts.binary_dump.size() > 0)
        saveCurrentAllBinary();
	exit(1);
}

int main(int argc, char *argv[])
{
	struct sigaction sig_int_handler;
	sig_int_handler.sa_handler = my_handler;
	sigemptyset(&sig_int_handler.sa_mask);
	sig_int_handler.sa_flags = 0;

	sigaction(SIGINT, &sig_int_handler, NULL);

    getProgramOptions(argc, argv, prog_opts);
    getDeviceInfo();
    if (prog_opts.device < 0)
        exit(0);

	/* XXX: Use mersenne twister */
	srand(time(NULL));

	readCnf(prog_opts.infile.c_str(), variables, reverse_variables, clauses);

    if (prog_opts.solution.size() != 0){
    	nr_variables = 1 + variables.size();
        nr_bit_words = 1 + (nr_variables >> 5);
        host_bit_values = new uint32_t[nr_bit_words];

        readSolution(prog_opts.solution.c_str(), variables, host_bit_values, nr_bit_words);
        printSolution(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, 0, true);

        delete[] host_bit_values;
        exit(0);
    }

	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	if (num_platforms == 0) {
		fprintf(stderr, "No OpenCL platforms available\n");
		exit(EXIT_FAILURE);
	}

	cl_platform_id platforms[num_platforms];
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	unsigned int platform;
	for (unsigned int i = 0; i < num_platforms; ++i) {
		size_t size;
		if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clGetPlatformInfo() returned failure\n");
			continue;
		}

		char name[size];
		if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, name, 0)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clGetPlatformInfo() returned failure\n");
			continue;
		}

		platform = i;
	}

	cl_uint num_devices;
	if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices)
		!= CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	cl_device_id devices[num_devices];
	if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL)
		!= CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	unsigned int device;
	for (unsigned int i = 0; i < num_devices; ++i) {
		size_t size;
		if (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clDeviceInfo() returned failure\n");
			continue;
		}

		char name[size];
		if (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, NULL)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clDeviceInfo() returned failure\n");
			continue;
		}

		device = i;
	}
	device = prog_opts.device;  // use selected device

	cl_int err;
	cl_context context = clCreateContext(NULL, 1, &devices[device], &pfn_notify, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateContext() failed\n");
		exit(EXIT_FAILURE);
	}

	cl_command_queue queue = clCreateCommandQueue(context, devices[device], 0, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateCommandQueue() failed\n");
		exit(EXIT_FAILURE);
	}

	int fd = open("kernel.cl", O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "open() failed\n");
		exit(EXIT_FAILURE);
	}

	struct stat st;
	if (fstat(fd, &st) == -1) {
		fprintf(stderr, "stat() failed\n");
		exit(EXIT_FAILURE);
	}

	void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (ptr == MAP_FAILED) {
		fprintf(stderr, "mmap() failed\n");
		exit(EXIT_FAILURE);
	}

	const char *string = (char *) ptr;
	const size_t length = st.st_size;

	cl_program program = clCreateProgramWithSource(context, 1, &string, &length, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram() failed\n");

		size_t size;
		if (clGetProgramBuildInfo(program, devices[device],
			CL_PROGRAM_BUILD_LOG, 0, NULL, &size) != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramBuildInfo() failed\n");
			exit(EXIT_FAILURE);
		}

		char log[size];
		if (clGetProgramBuildInfo(program, devices[device],
			CL_PROGRAM_BUILD_LOG, size, log, NULL) != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramBuildInfo() failed\n");
			exit(EXIT_FAILURE);
		}

		fprintf(stderr, "%s\n", log);
		exit(EXIT_FAILURE);
	}

	cl_kernel kernel = clCreateKernel(program, "search", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel() failed\n");
		exit(EXIT_FAILURE);
	}

	host_threads = new thread_state[prog_opts.nr_threads];

	for (unsigned int i = 0; i < prog_opts.nr_threads; ++i) {
		host_threads[i].rnd = rand();
	}

	cl_mem threads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, prog_opts.nr_threads * sizeof(*host_threads), host_threads, &err);
	if (err != CL_SUCCESS){
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////77

	nr_variables = 1 + variables.size();
	nr_bit_words = 1 + (nr_variables >> 5);
	host_bit_values = new uint32_t[nr_bit_words * prog_opts.nr_threads];

	if (prog_opts.intermediates.size() > 0){
        readIntermediates(prog_opts.intermediates.c_str(), variables, host_bit_values, nr_bit_words, prog_opts.nr_threads);
	} else if (prog_opts.binary_intermediates.size() > 0){
        if (prog_opts.binary_intermediates.find(" ") != std::string::npos){
            int index = prog_opts.binary_intermediates.find(" ");

            std::string file_name1 = prog_opts.binary_intermediates.substr(0, index);
            std::string file_name2 = prog_opts.binary_intermediates.substr(index+1, prog_opts.binary_intermediates.size());

            readIntermediatesBinary(file_name1.c_str(), host_bit_values, nr_bit_words, prog_opts.nr_threads);

            uint32_t *host_bit_values2 = new uint32_t[nr_bit_words * prog_opts.nr_threads];
            readIntermediatesBinary(file_name2.c_str(), host_bit_values2, nr_bit_words, prog_opts.nr_threads);

            // create infos about clauses for gpu
            struct object_info device_clauses_info;
            makeDeviceClauses(clauses, device_clauses_info);

            // create infos about variables to clauses mapping for gpu
            struct object_info device_variables_to_clauses_info;
            makeDeviceVariablesToClauses(clauses, nr_variables, device_variables_to_clauses_info);

            // ---------------------

            std::map<unsigned int, unsigned int> clause_collection;
            std::map<unsigned int, unsigned int> clause_collection2;

            printf("v trying to improve best individual ...\n");
            unsigned int blocks = 20;
            unsigned int percent = prog_opts.nr_threads/blocks;
            for (unsigned int j = 0; j < prog_opts.nr_threads; j++){

                if (j % percent == 0)
                    printf("v    %i%\n", (int)((100/blocks)*(j/percent)));

                //unsigned int before = 0;
                //for (unsigned int i = 0; i < clauses.size(); i++)
                //    if (!evaluate_clause_(device_clauses_info.address, i, host_bit_values, nr_bit_words, 0, 0))
                //        before++;

                for (unsigned int i = 0; i < clauses.size(); i++){
                    if (!evaluate_clause_(device_clauses_info.address, i, host_bit_values, nr_bit_words, 0, 0) && evaluate_clause_(device_clauses_info.address, i, host_bit_values2, nr_bit_words, 0, j)){
                        //printf("jetzt\t%i\n", i);
                        uint32_t index_clause = device_clauses_info.address[i];	        // get index where info of random clause is stored
                        uint32_t nr_lits = device_clauses_info.address[index_clause];	// first field contains number of literals

                        // collect the differing variables in clause
                        std::vector<unsigned int> to_flip;
                        for (unsigned int k = 0; k < nr_lits; k++){
                            uint32_t lit = device_clauses_info.address[index_clause + 1 + k];	// get literal
                            bool sign = lit & 1;	    // extract sign
                            uint32_t var = lit >> 1;	// extract variable

                            bool state1 = get_bit_(var, host_bit_values, nr_bit_words, 0);
                            bool state2 = get_bit_(var, host_bit_values2, nr_bit_words, j);

                            if (state1 != state2)
                                to_flip.push_back(var);
                        }

                        // get unsatisfied clauses connected with differing variables
                        clause_collection.clear();
                        for (unsigned int k = 0; k < to_flip.size(); k++)
                            evaluate_clauses_of_variable_unsatisfied2_(to_flip[k], device_variables_to_clauses_info.address, device_clauses_info.address, host_bit_values, nr_bit_words, false, 0, clause_collection);

                        // flip variables to change
                        for (unsigned int k = 0; k < to_flip.size(); k++)
                            flip_bit_(to_flip[k], host_bit_values, nr_bit_words, 0);

                        // get unsatisfied clauses connected with differing variables
                        clause_collection2.clear();
                        for (unsigned int k = 0; k < to_flip.size(); k++)
                            evaluate_clauses_of_variable_unsatisfied2_(to_flip[k], device_variables_to_clauses_info.address, device_clauses_info.address, host_bit_values, nr_bit_words, false, 0, clause_collection2);

                        if (clause_collection.size() < clause_collection2.size()){  // this works with "<" but NOT with "<="!!!
                            // undo changed variables
                            for (unsigned int k = 0; k < to_flip.size(); k++)
                                flip_bit_(to_flip[k], host_bit_values, nr_bit_words, 0);
                        } else {
                            int delta = clause_collection.size() - clause_collection2.size();
                            if (delta > 0)
                                printf("v       correcting clause %i, improved by %i\n", i, delta);
                        }
                    }
                }

                //unsigned int after = 0;
                //for (unsigned int i = 0; i < clauses.size(); i++)
                //    if (!evaluate_clause_(device_clauses_info.address, i, host_bit_values, nr_bit_words, 0, 0))
                //        after++;

                //printf("%i %i\n", before, after);
            }
            delete[] host_bit_values2;

            printf("v ... done\n");
        } else {
            readIntermediatesBinary(prog_opts.binary_intermediates.c_str(), host_bit_values, nr_bit_words, prog_opts.nr_threads);
        }
	} else {
		for (unsigned int j = 0; j < prog_opts.nr_threads; ++j){
			// variable 0 is always false, i.e., the most right bit is zero
			host_bit_values[j * nr_bit_words + 0] = (rand() >> 1) << 1;
			// fill the rest with random bytes
			for (unsigned int i = 1; i < nr_bit_words; ++i) {
				host_bit_values[j * nr_bit_words + i] = rand();
			}
		}
	}

    if (prog_opts.compare){
        compareClauses(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, prog_opts.nr_threads);
        exit(0);
    }

    if (prog_opts.show_intermediate > -1){
        printSolution(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, prog_opts.show_intermediate, false);
        exit(0);
    }

	cl_mem bit_values = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nr_bit_words * prog_opts.nr_threads * sizeof(*host_bit_values), host_bit_values, &err);
	if (err != CL_SUCCESS){
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	unsigned int nr_clauses = clauses.size();

	// create infos about clauses for gpu
	struct object_info device_clauses_info;
	makeDeviceClauses(clauses, device_clauses_info);

	cl_mem my_device_clauses = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, device_clauses_info.size*sizeof(*device_clauses_info.address), device_clauses_info.address, &err);
	if (err != CL_SUCCESS){
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	// create infos about variables to clauses mapping for gpu
	struct object_info device_variables_to_clauses_info;
	makeDeviceVariablesToClauses(clauses, nr_variables, device_variables_to_clauses_info);

	cl_mem my_device_variables_to_clauses = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, device_variables_to_clauses_info.size*sizeof(*device_variables_to_clauses_info.address), device_variables_to_clauses_info.address, &err);
	if (err != CL_SUCCESS){
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (clSetKernelArg(kernel, 0, sizeof(threads), &threads) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 1, sizeof(bit_values), &bit_values) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 2, sizeof(nr_bit_words), &nr_bit_words) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 3, sizeof(my_device_clauses), &my_device_clauses) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 4, sizeof(nr_clauses), &nr_clauses) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 5, sizeof(my_device_variables_to_clauses), &my_device_variables_to_clauses) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 6, sizeof(nr_variables), &nr_variables) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 7, sizeof(prog_opts.nr_iterations), &prog_opts.nr_iterations) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 8, sizeof(prog_opts.random_flips), &prog_opts.random_flips) != CL_SUCCESS){
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	struct timeval tv_a;
	if (gettimeofday(&tv_a, NULL) == -1) {
		fprintf(stderr, "gettimeofday() failed\n");
		exit(EXIT_FAILURE);
	}

    printf("v some infos ...\n");
	printf("v    number of variables: \t\t%i\n", nr_variables);
	printf("v    number of bit words: \t\t%i\n", nr_bit_words);
	printf("v    number of clauses: \t\t%i\n", nr_clauses);
    printf("v    size of clauses object: \t\t%i\n", device_clauses_info.size);
    printf("v    size of vars to clauses object: \t%i\n", device_variables_to_clauses_info.size);
	printf("v ... done\n");

	size_t work_offset = 0;
	size_t work_size = prog_opts.nr_threads;
	size_t local_work_size = prog_opts.nr_threads/prog_opts.nr_workgroups;

	uint32_t *host_bit_values_ = new uint32_t[nr_bit_words * prog_opts.nr_threads];

	while (1) {
		if ((err = clEnqueueNDRangeKernel(queue, kernel, 1,	&work_offset, &work_size,	&local_work_size, 0, NULL, NULL)) != CL_SUCCESS){
			printf("%d\n", err);

			fprintf(stderr, "clEnqueue() failed\n");
			exit(EXIT_FAILURE);
		}

		if (clEnqueueReadBuffer(queue, threads, CL_TRUE, 0,	prog_opts.nr_threads * sizeof(*host_threads), host_threads, 0, NULL, NULL) != CL_SUCCESS){
			fprintf(stderr, "clEnqueueReadBuffer() failed\n");
			exit(EXIT_FAILURE);
		}

		if (clFinish(queue) != CL_SUCCESS){
			fprintf(stderr, "clFinish() failed\n");
			exit(EXIT_FAILURE);
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		bool found_solution = false;
		unsigned int max_nr_sat_clauses = 0;
		unsigned int max_thread = -1;
		for (unsigned int i = 0; i < prog_opts.nr_threads; i++) {
			unsigned int nr_sat_clauses = host_threads[i].nr_satisfied;
			//printf(" nr_satisfied: %i\n", nr_sat_clauses);

			if (nr_sat_clauses > max_nr_sat_clauses){
				max_nr_sat_clauses = nr_sat_clauses;
				max_thread = i;
            }

			if (nr_sat_clauses < nr_clauses)
				continue;

			if (!found_solution) {
				// get buffers back from graphics card
				if (clEnqueueReadBuffer(queue, bit_values, CL_TRUE, 0, prog_opts.nr_threads * nr_bit_words * sizeof(*host_bit_values), host_bit_values, 0, NULL, NULL) != CL_SUCCESS){
					fprintf(stderr, "clEnqueueReadBuffer() failed\n");
					exit(EXIT_FAILURE);
				}

				if (clFinish(queue) != CL_SUCCESS){
					fprintf(stderr, "clFinish() failed\n");
					exit(EXIT_FAILURE);
				}
			}

			// We have a solution!
			found_solution = true;

			printSolution(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, i, true);
		}

		if (found_solution)
			break;

        unsigned int undecided = printSimplified(reverse_variables, nr_variables, clauses, host_bit_values, nr_bit_words, max_thread, false);
		printf("c best: %u %u\n", max_nr_sat_clauses, undecided);

		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		if (prog_opts.reuse > 0.0){
			if (prog_opts.reuse < 1.0){
				// get buffers back from graphics card
				if (clEnqueueReadBuffer(queue, bit_values, CL_TRUE, 0, prog_opts.nr_threads * nr_bit_words * sizeof(*host_bit_values), host_bit_values, 0, NULL, NULL) != CL_SUCCESS){
					fprintf(stderr, "clEnqueueReadBuffer() failed\n");
					exit(EXIT_FAILURE);
				}

				int nr_best = prog_opts.nr_threads*prog_opts.reuse;

				my_pair pairs[prog_opts.nr_threads];
				for (unsigned int i = 0; i < prog_opts.nr_threads; i++){
					pairs[i].first = i;
					pairs[i].second = host_threads[i].nr_satisfied;
				}

				std::sort(pairs, pairs + prog_opts.nr_threads, &pairSorterSecondDown);

				// copy best solutions to buffer
				for (unsigned int j = 0; j < prog_opts.nr_threads; j++){
					uint32_t index = pairs[j%nr_best].first;
					for (unsigned int i = 0; i < nr_bit_words; i++){
						host_bit_values_[j*nr_bit_words + i] = host_bit_values[index*nr_bit_words + i];
					}
				}
				// copy buffer back
				for (unsigned int j = 0; j < prog_opts.nr_threads; j++){
					uint32_t j1 = rand()%prog_opts.nr_threads;
					uint32_t j2 = rand()%prog_opts.nr_threads;
					for (unsigned int i = 0; i < nr_bit_words; i++){
						uint32_t word = host_bit_values_[j*nr_bit_words + i];
						if (prog_opts.crossover){
							uint32_t word1 = host_bit_values_[j1*nr_bit_words + i] & 0b10101010101010101010101010101010;
							uint32_t word2 = host_bit_values_[j2*nr_bit_words + i] & 0b01010101010101010101010101010101;
							word = word1 | word2;
						}
						host_bit_values[j*nr_bit_words + i] = word;
					}
				}

                clReleaseMemObject(bit_values);
				bit_values = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nr_bit_words * prog_opts.nr_threads * sizeof(*host_bit_values), host_bit_values, &err);
				if (err != CL_SUCCESS){
					fprintf(stderr, "clCreateBuffer() failed\n");
					exit(EXIT_FAILURE);
				}

				if (clSetKernelArg(kernel, 1, sizeof(bit_values), &bit_values) != CL_SUCCESS){
					fprintf(stderr, "clSetKernelArg() failed\n");
					exit(EXIT_FAILURE);
				}
			}
		} else {
			// get totally new starting values
			for (unsigned int j = 0; j < prog_opts.nr_threads; ++j){
				// variable 0 is always false, i.e., the most right bit is zero
				host_bit_values[j * nr_bit_words + 0] = (rand() >> 1) << 1;
				// fill the rest with random bytes
				for (unsigned int i = 1; i < nr_bit_words; ++i) {
					host_bit_values[j * nr_bit_words + i] = rand();
				}
			}

            clReleaseMemObject(bit_values);
			bit_values = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nr_bit_words * prog_opts.nr_threads * sizeof(*host_bit_values), host_bit_values, &err);
			if (err != CL_SUCCESS){
				fprintf(stderr, "clCreateBuffer() failed\n");
				exit(EXIT_FAILURE);
			}

			if (clSetKernelArg(kernel, 1, sizeof(bit_values), &bit_values) != CL_SUCCESS){
				fprintf(stderr, "clSetKernelArg() failed\n");
				exit(EXIT_FAILURE);
			}
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////////

	}

	struct timeval tv_b;
	if (gettimeofday(&tv_b, NULL) == -1) {
		fprintf(stderr, "gettimeofday() failed\n");
		exit(EXIT_FAILURE);
	}

	struct timeval tv_delta;
	timersub(&tv_b, &tv_a, &tv_delta);
	fprintf(stderr, "c Wall time: %lu.%06lu\n", tv_delta.tv_sec, tv_delta.tv_usec);

	delete[] host_threads;
	delete[] host_bit_values;
	delete[] host_bit_values_;

	clReleaseMemObject(threads);
	clReleaseMemObject(bit_values);
	clReleaseMemObject(my_device_clauses);
	clReleaseMemObject(my_device_variables_to_clauses);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
