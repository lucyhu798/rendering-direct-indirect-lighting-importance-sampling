#  Final Project Info
---
Contributers: Lucy Hu, Danny Vo

Emails: luhu@ucsd.edu, dmvo@ucsd.edu 

## Compilation Instructions
To compile, first make sure you have the required embree3 libraries. Run the following lines of code while in the repository to set up the project.

```
mkdir build
cd build
cmake ..
```

After finishing CMake configurations, run `make`.

Run the program using `./rt168` in the build folder. Only one file can be used as the input for the program.

## Running the Program
The program requires a .test file that contains parameters describing the scene. For this implementation, use the `circle.test` file found in `scenes/hw4/` that implements the features from the research article. More specifically, use the `mirror` directive to turn the functionality on.

Example execution:  
From the build folder, run the following:  
```./rt168 ../scenes/hw4/circle.test```

The output rendering file will be found in the file `circle.png`.