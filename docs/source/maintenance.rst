Maintenance
===========

Install the environment
-----------------------

This project is run with python, the best way to run our code is to generate a 
virtualenv and install the dependency as follows:

.. code-block:: bash

   # Create a virtualenv for the project
   virtualenv bathy_env
   source bathy_env/bin/activate
   pip -r requirements.txt 


Edit the documentation
----------------------

The documentation is built with sphinx. Its content is gathered in the source
folder of the ``docs``. You can update the documentation after any change by
starting the ``build_html.sh`` executable. Please note that if you add a 
``.rst`` file you need to link it into the tree structure, for example in the
toctree of the ``index.rst``.

.. code-block:: bash

   # Update the Documentation
   ./build_html.sh


Contribute to the project
-------------------------

The ``src`` folder contains all the modules of the project. If you want to make
your own module, please put a ``__init__.py`` file into your package and add it
to the ``src``. The ``bin`` and the ``tests`` folders contains all the scripts.
If you want to build such a script, don't forget to put the global PATH at the
root of the project by typing ``import context`` at the head of your file to
have access to the ``src`` packages or the ``dataset`` for example. I inform you
that the commenting of your code will be hardly judged and validated by myself 
:p


Use a dataset
-------------

At this stage of the project, you should give a path to the dataset to be
computed, this path should contains a folder named ``dataset`` with:
- ``train_TS``: contains the timestacks (:math:`600\times200` in our case).
- ``train_encoded_TS``: contains the encoded timestacks (:math:`100\times200` in our case).
- ``train_GT``: contains the bathymetries.

These names are hard-coded in our modules, if you don't fit this architecture,
it will certainly crash.

.. note:: 
        For our study case, this architecture was appropriate, but we are aware of
        the facti that it is not yet super flexible. For example, you may not need encoded
        data or other stuff, that's why we invite you to improve it and change the
        code if needed.

.. note::
        An example of dataset is already provided in the github repo.


Generated models
----------------

You may have noticed that the repo contains a folder named ``saves``, in that
folder we're saving our models, weights, and losses. Normally the architecture
of the models and the weights are automatically saved in the ``fit`` functions.
Actually the weights are saved at the end of each repeat of the ``fit``, in that
way you're able to regenerate a model at different stage of the fitting. You need
to call save ``save_losses`` at the end of the fit if you want to record the losses.


Write your own code
-------------------

We provide few examples of use case in the ``bin`` folder. They are technically ready to
work on the little dataset provided. You can find the API of the useful modules in this
documentation and create your own script to manipulate easily the learning on this
timestack problem.


GUI
---

This project also contains a GUI that you can launch with the script ``bin/window.py``.
This interface take a real timestack, its bathymetry, an encoder and a cnn model to
print the estimated bathymetry with the models selected.




