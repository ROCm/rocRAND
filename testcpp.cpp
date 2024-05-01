struct {
	int s : 4; //wrong: behavior of bit-field members with implicit signage vary across compilers
	unsigned int : 24; //correct: explicitly unsigned
	signed int : 4; //correct: explicitly signed
} bits;

//Function foo's array parameter has a specified size
void foo(int a[10]) {
	int i = 0;
	for (i = 0; i <10; i++) {
		a[i] = i * 2;
	}
}

int my_arr[5];
foo(my_arr); //my_arr is smaller than foo's array parameter, and will cause access to memory outside its bounds
