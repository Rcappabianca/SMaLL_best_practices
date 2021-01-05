.. _intro-to-github:

Introduction to Git and GitHub
==============================

* Fork the `repository`_.
* ``clone`` the forked repository localy.
* using ``python 3.5`` (or higher) install the minimum required library with the comand: 
   
   .. code-block:: console

      $ pip3 install -r SMALL_BEST_PRACTICES/docs/requirements.text

* Create a ``branch`` 

   .. code-block:: console

      $ git checkout -b your-new-branch-name

* Make your changes.
* Compile locacaly the *HTML* files and check that all is in the right place. To compile use the make script, and in the folder ``SMaLL_best_practices/docs/build/html`` you will find all the ``.html`` files:

   .. code-block:: console

      $ cd SMaLL_best_practices/docs
      $ ./make.bat html
   
* add your name to the file ``CREDITS.md``
* Push changes

   * add all the files

   .. code-block:: console

      $ git add .

   * commit those changes
   
   .. code-block:: console

      $ git commit -m "I make this changes: <changes>"

   * push your changes 
   
   .. code-block:: console

      $ git push origin your-new-branch-name

* Submit your changes for review

.. toctree::
   :maxdepth: 2
   :hidden:
