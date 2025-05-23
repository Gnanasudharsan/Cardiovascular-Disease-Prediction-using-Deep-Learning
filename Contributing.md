# Contributing to Cardiovascular Disease Prediction

We welcome contributions to improve this cardiovascular disease prediction project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check existing issues** first to avoid duplicates
2. **Use a clear title** that describes the bug
3. **Provide detailed information**:
   - Operating system and version
   - Python version
   - Library versions
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages or logs

### Suggesting Enhancements

1. **Check if the enhancement is already suggested**
2. **Provide a clear title** and detailed description
3. **Explain why this enhancement would be useful**
4. **Consider implementation details** if possible

### Code Contributions

#### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/cardiovascular-disease-prediction.git
   cd cardiovascular-disease-prediction
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

#### Making Changes

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Add tests** for new functionality

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

5. **Update documentation** if needed

#### Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Code Formatting**: Use `black` for automatic formatting:
  ```bash
  black src/ tests/ main.py
  ```
- **Linting**: Use `flake8` for linting:
  ```bash
  flake8 src/ tests/ main.py
  ```
- **Documentation**: Use docstrings for all functions and classes
- **Type Hints**: Add type hints where appropriate

#### Pull Request Process

1. **Update your branch** with the latest changes:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - List of changes made
   - Screenshots if applicable

4. **Address review feedback** promptly

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Test both normal and edge cases
- Mock external dependencies

Example test structure:
```python
def test_feature_selection_with_valid_data():
    """Test feature selection with valid input data."""
    # Arrange
    X, y = create_sample_data()
    selector = FeatureSelector(config)
    
    # Act
    X_selected = selector.fit_transform(X, y)
    
    # Assert
    assert X_selected.shape[1] < X.shape[1]
    assert len(selector.selected_features_) > 0
```

## 📊 Research Contributions

We especially welcome contributions that:

- **Improve model performance** while maintaining interpretability
- **Add new evaluation metrics** relevant to healthcare
- **Enhance data preprocessing** techniques
- **Provide new visualizations** for medical professionals
- **Add support for new datasets** or data formats
- **Improve computational efficiency**

### Research Standards

- **Reproducibility**: Ensure experiments can be reproduced
- **Documentation**: Provide detailed methodology descriptions
- **Validation**: Use proper cross-validation techniques
- **Comparison**: Compare against existing baselines
- **Statistical significance**: Report confidence intervals and significance tests

## 📚 Documentation

### Updating Documentation

- Update `README.md` for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Update configuration documentation

### Writing Style

- Use clear, concise language
- Provide practical examples
- Include code snippets where helpful
- Consider non-expert users

## 🐛 Issue Labels

We use the following labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `research`: Related to research methodology
- `performance`: Performance improvements

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

1. **Model Improvements**:
   - Alternative deep learning architectures
   - Ensemble methods
   - Hyperparameter optimization

2. **Data Engineering**:
   - Data validation and quality checks
   - Missing data handling techniques
   - Feature engineering methods

3. **Evaluation & Metrics**:
   - Clinical performance metrics
   - Model interpretability tools
   - Bias and fairness analysis

4. **Deployment & Production**:
   - Model serving capabilities
   - API development
   - Performance monitoring

5. **Healthcare Integration**:
   - FHIR compatibility
   - Electronic Health Record integration
   - Clinical decision support tools

## 📋 Checklist for Contributors

Before submitting your contribution, ensure:

- [ ] Code follows project style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] Changes are described in pull request
- [ ] No sensitive data is included
- [ ] Dependencies are properly declared
- [ ] Code is well-commented
- [ ] Performance impact is considered

## 🤔 Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Reach out to maintainers

## 📝 Code of Conduct

Please note that this project follows a Code of Conduct. By participating, you are expected to uphold this code:

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Focus on what's best** for the community
- **Show empathy** towards other community members

## 🙏 Recognition

Contributors will be recognized in:

- Repository contributors list
- Release notes for significant contributions
- Research paper acknowledgments (for research contributions)

Thank you for contributing to improving cardiovascular disease prediction through AI! 🚀

---

For more information, see our [README.md](README.md) or contact the maintainers.
