# AMACS

This repo is AMACS, which stands for **Against Matlab at CentraleSupélec**.
We understand that Matlab is a powerful tool, today used in many industries. We think students
should be familiar with Matlab thanks to the appropriate training courses offered at CentraleSupélec.

However, Matlab is a *proprietary language*, and almost all its features are now available in Python, an *open source language*.
We thus think **students should have the choice** and the ability to use Python for the optimization course. This was not the case, and this directory solves this problem by providing a python translation of all the tutorial Matlab files.

**Don't hesitate to add a star if this repo has helped you**

<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/alexgravx/AMACS?style=social&logoColor=yellow&label=Give%20a%20star">

# Tutorials overview

- TD1 - Convex Optimization: application on digital images recovery
- TD2 - Linear Programming: maximize the space covered by a vacuum robot
- TD3 - Integer Linear Programming: Bin Packing problem with items and boxes
- TD4 - Constraints in continous optimization: application to Euromillion prize

# Quick start

## MacOS / Linux

Note: 
- You'll need *pyenv* installed, see here: https://github.com/pyenv/pyenv
- You'll also need *make* command installed, via brew on MacOS or apt on Linux

First, setup the env with:

````
make setup
make install
````

Then, for each tutorial, there is a associated folder (TD1, TD2, TD3 and TD4).
In each folder, you will find `subject/` and `correction/` subfolders.
They contain python files needed for the tutorial, as well as a *Jupyter Notebook* which is **the main file to fill to complete the programming part of the tutorial.**
If you are using VSCode, don't forget to select the right virtual environment for the jupyter notebook.

You can last delete the env with:

````
make delete
````

## Windows

You can configure a virtual environment with the packages provided in `requirements.txt`. \
Install them with `pip install -r requirements.txt`

# Authors

- Alexandre Gravereaux
- Hugues du Moulinet D'Hardemare

# Contributions

Licence Creative Commons CC BY-SA

This repo is open to all contributions, whether for improvements or updates as TD topics evolve.

Don't forget to **cite** all the authors who contributed to this work.
