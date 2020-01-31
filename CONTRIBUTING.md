# How to contribute

Thank you for your interest in **afinidata**! We are quite glad of you reading this, as **afinidata** is an open project, and while it is not actively looking for volunteers, your support is quite appreciated. Please read throughly to ensure your contributions are the most effective.

**afinidata** is built using several components, one of which is the content recommender. Our recommendations are based on the performance a child has had on past activities. We choose activities in which we can predict that the child will perform well from a development area in which she has not excelled in average. Technically, our prediction for performance is based on machine learning techniques, namely a collaborative filtering model with explicit mean and item and user biases. More about the model and the recommendation strategy can be found in the [README](README.md).

Here are some important resources:

  * [Afinidata Home Page](http://afinidata.com/) tells you what we are,
  * [Our Task list](#) Currently our tasks and roadmap are still run privately, we'll update it if it changes.
  * [Issue / Bug Tracker](https://github.com/afinidata2019/afinidata_recommender/issues).
  
## Running in development

To run **afinidata** on development:

    $ git clone https://github.com/afinidata2019/afinidata_recommender.git
    $ python install -r requirements.txt
    
## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to coc@afinidata.com

## Testing

We use pytest to write tests. These tests cover particular steps of the whole model definition: data preparation and test and train splitting, loss minimization during model training and predictions. Please add at least a test for each feature in order to avoid lowering of our testing coverage score.

## Submitting changes

Please send a [GitHub Pull Request](https://github.com/afinidata2019/afinidata_recommender/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). When you send a pull request, please detail what and why is it useful. Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit). If not it might be squashed and merged.

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."
    
## Documenting

We use sphinx to autogenerate documentation. It is advised to at least add docstrings to new member functions or classes.

## Coding conventions

Start reading our code and you'll get the hang of it. We try to optimize for readability but we do have some pain points due to our legacy codebase.

  * Indent using two spaces (soft tabs)
  * We follow PEP-8 loosely, periodically run a linter to ensure uniform code formatting.
  * This is open source software. Consider the people who will read your code, and make it look nice for them. It's sort of like driving a car: Perhaps you love doing donuts when you're alone, but with passengers the goal is to make the ride as smooth as possible.
  * And More, pull requests regarding this, contributing or documentation are also quite appreciated


on behalf of **afini**'s team 
thanks again for your interest,
Pedro, Data scientist @ Afinidata