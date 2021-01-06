.. Links:

.. _GitHub: https://github.com
.. _repository website: https://github.com/DAP93/SMaLL_best_practices
.. _Sphinx page: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html

.. _intro-to-github:

Introduction to Git and GitHub
==============================

Git
---

`Git` is a distributed version control system designed to track changes in any set of files. 
It is a multiplatform program and is usually integrated into different modern text-editors.

.. important::
   The following commands assume you are in a *Linux environment* however, the same commands 
   can be used in *Windows PowerShell*


Installing on Windows
^^^^^^^^^^^^^^^^^^^^^

#. Get last  :download:`git <https://git-scm.com/download/win>` installation file.
#. Execute the file ``Git-x.xx.x-64-bit.exe``
#. Follow the automated installation. 

Installing on Linux
^^^^^^^^^^^^^^^^^^^

Get ``Git`` from your Linux distribution repository (e.g. ``apt``).

.. code-block:: bash

   $ sudo apt install git-all

Installing on macOS
^^^^^^^^^^^^^^^^^^^

There is several ways to install Git on a Mac. The easiest is probably to install the Xcode Command Line Tools 
and try to run git from the Terminal.

.. code-block:: bash

   $ git --version

If you do not have it installed already, it will prompt you to install it.


GitHub
------

GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.
GitHub and similar services (including GitLab and BitBucket) are websites that host a Git server program to hold your code. 
Thus using ``git`` you can communicate with the remote hosting services.

Create a GitHub account
^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to get started is to create an account on `GitHub`_ website (it is free).

.. _github_fig1:
.. figure:: ../img/github_singup.png
         :width: 70 %
         :alt: github singup
         :align: center

         GitHub homepage.

Fork the repository
^^^^^^^^^^^^^^^^^^^

Go to the source code :fa:`external-link` `repository website`_ and by clicking on the fork button on the top of the page
(step 1 at :numref:`github_fig2`). 
This will create a copy of this repository in your account.

.. _github_fig2:
.. figure:: ../img/github_fork_and_clone.png
         :width: 70 %
         :alt: github singup
         :align: center

         Fork and clone the repository.

Clone the repository
^^^^^^^^^^^^^^^^^^^^

Now clone the forked repository to get a copy on your local machine using the command ``git clone``.
Go to your GitHub account, open the forked repository, click on the green code button, and then click the copy to clipboard icon (step 2 and 3 at :numref:`github_fig2`). And in the terminal type:

.. code-block:: bash

   $ git clone https://github.com/<your_username>/SMaLL_best_practices.git

Where you have to replace ``<your_username>`` with your GitHub username.

Create a Branch
^^^^^^^^^^^^^^^^

Branching is the way to work on different versions of a repository at one time (:numref:`github_fig3`).
To create a ``branch`` first, you have to move to the repository directory on your computer:

.. code-block:: bash

   $ cd SMaLL_best_practices

Now create a ``branch`` using the ``git checkout`` command:

.. code-block:: bash

   $ git checkout -b <your-new-branch-name>

Where ``<your-new-branch-name>`` is the name of the ``branch``.

.. _github_fig3:
.. figure:: ../img/github_branching.png
         :width: 90 %
         :alt: github singup
         :align: center

         Create, submit and merge a branch named ``feature``.

.. important::
   The branch name as to be a single word, thus replace the spaces with the underscore ``_`` 

.. note::
   The name of the branch does not need to have the word add in it, 
   but it's a reasonable thing to include because the purpose of this branch is to add your name to a list 
   and its main changes. 

Setup python environment
^^^^^^^^^^^^^^^^^^^^^^^^

Now that you have all the documentation in your local machine, you can have to install the minimum required python packages.

.. code-block:: bash

   $ pip3 install -r SMaLL_best_practices/docs/requirement.txt

We suggest creating a *virtual python environment* outside the repository folder (see :ref:`Python basics<python_basics>`).

.. important::
   If the *virtual python environment* is inside the repository folder, once you will ``push``(upload) your chage, you 
   will upload the virtual environment too.


Make your changes and commit those changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you can start to make your changes; the file format to use is the reStructuredText (``.rst``), basic concepts is available
in the section :ref:`reStructuredText`, for a more exhaustive guide, also read the `Sphinx page`_.

To check your changes, respect the old documentation, you can run the command ``git status``.

Once you finish modifying a page, you can upload it to GitHub with the command ``git commit``.
First, you add the changed files to the list of the commit:

.. code-block:: bash

   $ git add <file name>

or all the folder

.. code-block:: bash

   $ git add .

Then you can define a commit associated with the selected files.

.. code-block:: bash

   $ git commit -m "<comment>"

Example:

.. code-block:: bash

   $ git add ./doc/source/exmaple.rst
   $ git commit -m "add the page ./doc/source/exmaple.rst"

.. note::
   You can think of the ``git commit`` command as a special way to save your progress since instead to have many changes in only one
   *commit*, you brake it into smaller parts

Compile locally
^^^^^^^^^^^^^^^

To see the changes, you have to ways:

#. Using *Visual Studio Code* and the *reStructuredText* extension (or similar) to get a preview of the web page (see the page :ref:`visual_studio_code`)
#. Compile the documentation with the command ``make HTML``

   .. code-block:: bash

      $ make html          # linux/macos
      > .\make.bat html    # windows


If there are no errors, you will get something similar to what you see in :numref:`github_fig4`.

.. _github_fig4:
.. figure:: ../img/make_output.png
         :width: 70 %
         :alt: make html output
         :align: center

         ``make HTML`` output.


Then in the subfolder ``./docs/build/HTML``, you have all the HTML files, and you can see how your changes will appear in the website.

.. note::
   Try to solve all the errors or warnings, and compile more time untile it prints a clean output.

Push changes to GitHub
^^^^^^^^^^^^^^^^^^^^^^

Push (upload) your changes using the command ``git push``:

.. code-block:: bash

   $ git push origin <add-your-branch-name>

Replace ``<add-your-branch-name>`` with the name of the branch you created earlier.

Submit your changes for review
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you go to your repository on GitHub, you will see a ``Compare & pull request`` button. Click on that button.

.. _github_fig5:
.. figure:: ../img/github_pull_request1.png
         :width: 70 %
         :alt: pull request button
         :align: center

         ``Compare & pull request`` button.

Now submit the pull request.

.. _github_fig6:
.. figure:: ../img/github_pull_request2.png
         :width: 70 %
         :alt: submit pull request
         :align: center

         submit pull request screen.

Soon we will be reviewing and merging all your changes into the ``master branch``. 
You will get a notification email once the changes have been merged.



.. toctree::
   :maxdepth: 3
   :hidden:

