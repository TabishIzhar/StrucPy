Installation
=============

You will need Python and a few packages as dependency for StrucPy on your system.

Dependency
-------------------------------
* **numpy**: For solving dense matrix and calculation
* **pandas**: For taking input and presenting output
* **plotly**: For visualization and plotting
* **ray**: For multi-processing
* **openpyxl**: To read Excel files



Install the StrucPy
-------------------------------
You can install StrucPy with pip! It will take care of all dependencies and their versions.

.. code-block:: python

   py -3 -m pip install StrucPy

In case you need a specific version of the package, that’s possible too. Simple declare the version condition over the code in terminal.

.. code-block:: python

   py -3 -m pip install StrucPy==0.0.1



Clone from Git repository
-------------------------------
Alternatively, you can build the package from the source by cloning the source from the git repository. If you’d like to contribute to the development of `StrucPy`, then install from github.

1. Clone the repository using **https://github.com/TabishIzhar/StrucPy.git**.

2. Form a virtual environment using 

.. code-block::python

   py -3 -m venv venvStrucPy

3. Activate virtual environment from cmd

.. code-block:: python

   .\venv\Scripts\activate.bat

4. Install every dependency using requirement.txt

.. code-block:: python

   pip install -r requirements.txt




