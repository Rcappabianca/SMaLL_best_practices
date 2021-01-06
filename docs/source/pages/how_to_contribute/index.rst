.. Links:

.. _GitHub account: https://github.com
.. _repository: https://github.com/DAP93/SMaLL_best_practices

How to contribute
=================

All the documentation is stored on the GitHub page https://github.com/DAP93/SMaLL_best_practices. 
In this section, you will find some base concepts that will help you to contribute easily to the documentation.

To sum up, every time you want to add/modify something, you need to have a `GitHub account`_ and 
to follow the steps:

#. Fork the `repository`_.
#. ``clone`` the forked repository locally.
#. using ``python 3.5`` (or higher), install the minimum required libraries with the command: 
#. Create a ``branch``. 
#. Make your changes.
#. Compile the *HTML* files locally and check that all is in the right place. To compile it, use the ``make`` script, and in the folder ``SMaLL_best_practices/docs/build/html`` you will find all the ``.html`` files.
#. Add your name to the file ``CREDITS.md``.
#. Push the changes:

   * add all the files
   * commit those changes
   * push your changes

#. Submit your changes for review.

.. important::
   When you submit the branch to the review, please add some comments to allow the reader to understand what you did.

If you need a more step by step guide, check the subsection :ref:`intro-to-github`.

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: More info:

   ./sections/github
   ./sections/restructuredtext
   ./sections/visual_studio_code
