Maintenance
===========

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
have access to the ``src`` packages or the ``data`` for example. I inform you
that the commenting of your code will be hardly judged and validated by myself 
:p
