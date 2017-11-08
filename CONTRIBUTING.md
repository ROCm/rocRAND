# Contributing to rocRAND

## Contribution License Agreement
1. The code I am contributing is mine, and I have the right to license it.

2. By submitting a pull request for this project I am granting you a license to distribute said code under the MIT License for the project.

## How to contribute

Our code contriubtion guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).  This repository follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.
  * A [git extention](https://github.com/nvie/gitflow) has been developed to ease the use of the 'git flow' methodology, but requires manual installation by the user.  Refer to the projects wiki.

## Pull-request guidelines
* target the **develop** branch for integration
* ensure code builds successfully
* do not break existing test cases
* new functionality will only be merged with new unit tests
  * new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md)
  * tests must have good code coverage
  * code must also have benchmark tests, and performance must be acceptable to maintainers

