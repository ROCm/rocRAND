# Contributing to rocRAND

## Pull-request guidelines
* target the **develop** branch for integration
* ensure code builds successfully
* do not break existing test cases
* new functionality will only be merged with new unit tests
  * new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md)
  * tests must have good code coverage
  * code must also have benchmark tests, and performance must be acceptable to maintainers

