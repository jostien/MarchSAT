/* MarchSAT, Copyright (c) 2016 Jost Neigenfind <jostie@gmx.de>
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

// test

#define LOCAL_SIZE_0 256

typedef uchar u8;
typedef ushort u16;
typedef uint u32;
typedef ulong u64;

struct thread_state {
	u64 rnd;
	u32 nr_satisfied;
};

//u32 rnd(u32 current_rnd){
//	u32 ret = current_rnd ^ (((current_rnd << 4) | (current_rnd >> 28)) + 0xcafebabe);
//	return ret;
//}

u64 rnd(u64 current_rnd){
	u64 ret = (current_rnd * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	return ret;
}

bool get_bit(u32 var, __global u32 *bit_values, u32 nr_bit_words)
{
	u32 index = var >> 5;
    u32 offset = var % 32;

	u32 word = bit_values[index + nr_bit_words*get_global_id(0)];

	return (word << (31 - offset)) >> 31;
}

void flip_bit(u32 var, __global u32 *bit_values, u32 nr_bit_words)
{
	u32 index = var >> 5;
    u32 offset = var % 32;

	u32 word = bit_values[index + nr_bit_words*get_global_id(0)];
	word ^= 1 << offset;

	bit_values[index + nr_bit_words*get_global_id(0)] = word;
}

bool evaluate_literal(u32 lit, __global u32 *bit_values, u32 nr_bit_words, u32 flip_var)
{
	bool sign = lit & 1;
	u32 var = lit >> 1;

	u32 index = var >> 5;
    u32 offset = var % 32;

	u32 word = bit_values[index + nr_bit_words*get_global_id(0)];
	u32 evaluation = (word << (31 - offset)) >> 31;

	bool flip = var == flip_var;
	return (evaluation ^ sign) ^ flip;
}

bool evaluate_clause(__global const u32* clauses, u32 index_to_clause, __global u32 *bit_values, u32 nr_bit_words, u32 flip_var)
{
	// this function behaved weird and therefore this strange return pattern
	// there was a bug in one of the calling functions, might have been due to that

	u32 index 	= clauses[index_to_clause];		// get pointer to clause info
	u32 nr_lits = clauses[index];				// first field contains number of literals

	for (int i = 0; i < nr_lits; i++){
		u32 lit = clauses[index + 1 + i];
		if(evaluate_literal(lit, bit_values, nr_bit_words, flip_var))
			return true;
	}
	return false;
}

u32 evaluate_clauses_of_variable_unsatisfied(u32 var, __global const u32* variables_to_clauses, __global const u32* clauses, __global u32* bit_values, u32 nr_bit_words, bool flip){
	u32 index = variables_to_clauses[var]; 			// get index where variable to clauses info is stored
	u32 nr_clauses = variables_to_clauses[index];	// first field contains number of clauses the variable occurs in

	u32 evaluation = 0;
	for (unsigned int i = 0; i < nr_clauses; i++){
		u32 index_to_clause = variables_to_clauses[index + 1 + i];
		if (flip)
			evaluation += !evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, var);
		else
			evaluation += !evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, 0);
	}

	return evaluation;
}

u32 evaluate_clauses_of_variable_satisfied(u32 var, __global const u32* variables_to_clauses, __global const u32* clauses, __global u32* bit_values, u32 nr_bit_words, bool flip){
	u32 index = variables_to_clauses[var]; 			// get index where variable to clauses info is stored
	u32 nr_clauses = variables_to_clauses[index];	// first field contains number of clauses the variable occurs in

	u32 evaluation = 0;
		for (unsigned int i = 0; i < nr_clauses; i++){
		u32 index_to_clause = variables_to_clauses[index + 1 + i];
		if (flip)
			evaluation += evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, var);
		else
			evaluation += evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, 0);
	}

	return evaluation;
}

int find_clause(u32 index_to_clause, __local u32 *unsatisfied_clauses, u32 nr_unsatisfied){
    for (unsigned int i = 0; i < nr_unsatisfied; i++)
        if (unsatisfied_clauses[i] == index_to_clause)
            return i;

    return -1;
}

u32 add_unsatisfied_clause(u32 index_to_clause, __local u32 *unsatisfied_clauses, u32 nr_unsatisfied){
    // here still must be checked if new size does not violate buffer size
    unsatisfied_clauses[nr_unsatisfied] = index_to_clause;
    return nr_unsatisfied + 1;
}

u32 remove_unsatisfied_clause(u32 index_to_clause, __local u32 *unsatisfied_clauses, u32 nr_unsatisfied){
    int index = find_clause(index_to_clause, unsatisfied_clauses, nr_unsatisfied);
    unsatisfied_clauses[index] = unsatisfied_clauses[nr_unsatisfied - 1];
    return nr_unsatisfied - 1;
}

u32 evaluate_clauses_of_variable_unsatisfied2(u32 var, __global const u32* variables_to_clauses, __global const u32* clauses, __global u32* bit_values, u32 nr_bit_words, __local u32 *unsatisfied_clauses, u32 nr_unsatisfied){
	u32 index = variables_to_clauses[var]; 			// get index where variable to clauses info is stored
	u32 nr_clauses = variables_to_clauses[index];	// first field contains number of clauses the variable occurs in

    u32 counter = nr_unsatisfied;
	for (unsigned int i = 0; i < nr_clauses; i++){
		u32 index_to_clause = variables_to_clauses[index + 1 + i];
        bool flipped = evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, var);
		bool unflipped = evaluate_clause(clauses, index_to_clause, bit_values, nr_bit_words, 0);
		if (!flipped && unflipped){     // this is a new unsatisfied clause => add it
            counter = add_unsatisfied_clause(index_to_clause, unsatisfied_clauses, counter);
		}
		if (flipped && !unflipped){     // this is a new satisfied clause => remove it
            counter = remove_unsatisfied_clause(index_to_clause, unsatisfied_clauses, counter);
		}
		if (flipped && unflipped){      // both cases are satisfied => nothing must be added
		}
		if (!flipped && !unflipped){    // both cases are unsatisfied, already in list => nothing must be added
		}
	}

	return counter;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool evaluate_clause_t(__global const u32* clauses, u32 index_to_clause, __global u32 *bit_values, u32 nr_bit_words, u32 flip_var, u32 t)
{
    // is equivalent to counting clauses going from unsatisfied to satisfied if t == 1

        u32 index       = clauses[index_to_clause];             // get pointer to clause info
        u32 nr_lits = clauses[index];                           // first field contains number of literals

    bool state = false;
    u32 c = 0;
        for (int i = 0; i < nr_lits; i++){
                u32 lit = clauses[index + 1 + i];

                bool b = evaluate_literal(lit, bit_values, nr_bit_words, 0);  // get current evaluation

        if (flip_var == (lit >> 1))             // if variable to flip is current variable
            state = b;                          // then save state of variable to flip

                if(b)                                   // count all evaluations being true
                        c++;
        }

        if (!state)                                 // if variable to flip evaluates to false, then it does not contribute to c
        return c + 1 == t;                      // therefore, flipping would increase number of satisfied literals

    return c - 1 == t;                          // else flipping would decrease number of satisfied literals
}

u32 evaluate_clauses_of_variable_make_t(u32 var, __global const u32* variables_to_clauses, __global const u32* clauses, __global u32* bit_values, u32 nr_bit_words, u32 t){
        u32 index = variables_to_clauses[var];                  // get index where variable to clauses info is stored
        u32 nr_clauses = variables_to_clauses[index];   // first field contains number of clauses the variable occurs in

        u32 evaluation = 0;
        for (unsigned int i = 0; i < nr_clauses; i++){
                u32 index_to_clause = variables_to_clauses[index + 1 + i];
        evaluation += evaluate_clause_t(clauses, index_to_clause, bit_values, nr_bit_words, var, t);
        }

        return evaluation;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void search(__global struct thread_state *states,
	__global u32 *bit_values,
	unsigned int nr_bit_words,
	__global const u32 *clauses,
	unsigned int nr_clauses,
	__global const u32 *variables_to_clauses,
	unsigned int nr_variables,
	unsigned int n,
	double p)
{
    u32 biggest = 0;
    biggest = ~biggest;

    __local u32 unsatisfied_clauses[512];

    struct thread_state state = states[get_global_id(0)];

    int nr_satisfied = 0;
    for (unsigned int i = 0; i < nr_clauses; i++)
        if (evaluate_clause(clauses, i, bit_values, nr_bit_words, 0))
            nr_satisfied++;

    int nr_unsatisfied = nr_clauses - nr_satisfied;
    int nr_unsatisfied_old = nr_unsatisfied;

    // some space left if number of unsatisfied clauses increases
    if (nr_unsatisfied <= 448){
        // fill array with indices of unsatisfied clauses
        int counter = 0;
        for (unsigned int i = 0; i < nr_clauses; i++){
            if (!evaluate_clause(clauses, i, bit_values, nr_bit_words, 0)){
                unsatisfied_clauses[counter] = i;
                counter++;
            }
        }
    }

    for (int i = 0; i < n; i++){
        if(nr_unsatisfied > 0){
            // if there are too many unsatisfied clauses
            u32 clause_i;
            if (nr_unsatisfied > 448){
                // get random clause
                state.rnd = rnd(state.rnd);
                clause_i = state.rnd % nr_clauses;
            } else {
                // get random clause from list
                state.rnd = rnd(state.rnd);
                clause_i = unsatisfied_clauses[state.rnd % nr_unsatisfied];
            }

            // this should be satisfied, if nr_unsatisfied <= 448
            if (!evaluate_clause(clauses, clause_i, bit_values, nr_bit_words, 0)){
                u32 index_clause = clauses[clause_i];	// get index where info of random clause is stored
                u32 nr_lits = clauses[index_clause];	// first field contains number of literals

                // greedy or random
                state.rnd = rnd(state.rnd);
                u32 flip = state.rnd % biggest;
                if (flip <= p*biggest){
                    // which bit to randomyl flip
                    state.rnd = rnd(state.rnd);
                    u32 which_bit = state.rnd % nr_lits;

                    u32 lit = clauses[index_clause + 1 + which_bit];	// get literal
                    bool sign = lit & 1;	// extract sign
                    u32 var = lit >> 1;		// extract variable

                    int before = evaluate_clauses_of_variable_unsatisfied(var, variables_to_clauses, clauses, bit_values, nr_bit_words, false);
                    int after  = evaluate_clauses_of_variable_unsatisfied(var, variables_to_clauses, clauses, bit_values, nr_bit_words, true);

                    // if there are only a few unsatisfied clauses left, update list
                    if (nr_unsatisfied <= 448)
                        evaluate_clauses_of_variable_unsatisfied2(var, variables_to_clauses, clauses, bit_values, nr_bit_words, unsatisfied_clauses, nr_unsatisfied);

                    flip_bit(var, bit_values, nr_bit_words);	// flip the corresponding bit

                    nr_unsatisfied = nr_unsatisfied - before + after;
                } else{
                    // find literal, which flipped produces the minimal number of unsatsified clauses
                    int nr_unsatisfied_before = -1;
                    int nr_unsatisfied_after = -1;
                    u32 var_after = 0;
                    int lmk_score = -1;
                    for (unsigned int j = 0; j < nr_lits; j++){
                        u32 lit = clauses[index_clause + 1 + j];	// get literal
                        bool sign = lit & 1;	// extract sign
                        u32 var = lit >> 1;		// extract variable

                        int before = evaluate_clauses_of_variable_unsatisfied(var, variables_to_clauses, clauses, bit_values, nr_bit_words, false);
                        int after  = evaluate_clauses_of_variable_unsatisfied(var, variables_to_clauses, clauses, bit_values, nr_bit_words, true);
                        if (after <= before){
                            if (nr_lits > 3){
                                // if there are ties, then break them with lmake
                                if (nr_unsatisfied_after == -1 || after == nr_unsatisfied_after){
                                    int make1 = evaluate_clauses_of_variable_make_t(var, variables_to_clauses, clauses, bit_values, nr_bit_words, 1);
                                    int make2 = evaluate_clauses_of_variable_make_t(var, variables_to_clauses, clauses, bit_values, nr_bit_words, 2);
                                    int score = (nr_lits - 2)*make1 + (nr_lits - 3)*make2;
                                    if (lmk_score == -1 || score > lmk_score){
                                        lmk_score = score;
                                        var_after = var;
                                        nr_unsatisfied_before = before;
                                        nr_unsatisfied_after = after;
                                    }
                                }
                            }

                            // this we do always if we find literal with strictly less unsatisfied clauses
                            if (nr_unsatisfied_after == -1 || after < nr_unsatisfied_after){
                                lmk_score = -1;
                                var_after = var;
                                nr_unsatisfied_before = before;
                                nr_unsatisfied_after = after;
                            }
                        }
                    }
                    if (nr_unsatisfied_after != -1){
                        // if there are only a few unsatisfied clauses left, update list
                        if (nr_unsatisfied <= 448)
                            evaluate_clauses_of_variable_unsatisfied2(var_after, variables_to_clauses, clauses, bit_values, nr_bit_words, unsatisfied_clauses, nr_unsatisfied);

                        nr_unsatisfied = nr_unsatisfied - nr_unsatisfied_before + nr_unsatisfied_after;
                        flip_bit(var_after, bit_values, nr_bit_words);	// flip the corresponding bit
                    }
                }
            }
	}
        if (nr_unsatisfied < nr_unsatisfied_old)
            break;
    }

    state.nr_satisfied = nr_clauses - nr_unsatisfied;
    states[get_global_id(0)] = state;
}
