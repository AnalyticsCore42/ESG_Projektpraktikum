# Git Notes for fastai Contributors

## Setup and Basics

### Configure Git

First time setup:

```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

You might also want to configure the line ending preferences:

```
# On Unix/Mac/Linux
git config --global core.autocrlf input

# On Windows
git config --global core.autocrlf true
```

### Clone the Repository

```
git clone https://github.com/fastai/fastai.git
cd fastai
```

### Setup the Environment

```
conda env create -f environment.yml
conda activate fastai
pip install -e ".[dev]"
```

## Contributing Workflow

### 1. Sync Your Fork

```
# Add the main repository as upstream (first time only)
git remote add upstream https://github.com/fastai/fastai.git

# Sync your fork
git fetch upstream
git checkout master
git merge upstream/master
git push  # push to your fork
```

### 2. Create a Branch

```
git checkout -b my-new-feature
```

### 3. Make Changes and Commit

```
# Edit files...
git add <files changed>
git commit -m "Meaningful commit message"
```

### 4. Push Branch to GitHub

```
git push -u origin my-new-feature
```

### 5. Submit a Pull Request

Go to GitHub and create a Pull Request from your branch.

## Commit Message Guidelines

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add support for PyTorch 1.x tensors

This adds compatibility for the new PyTorch API while maintaining
backward compatibility.

Fixes #123
```

## Git Tips

### Amending Commits

If you need to modify your last commit:

```
git add <files to add>
git commit --amend
```

### Rebasing

To rebase your branch on the latest master:

```
git checkout master
git pull
git checkout my-feature-branch
git rebase master
```

### Interactive Rebase 

To clean up your branch before submitting a PR:

```
git rebase -i HEAD~3  # Rebase the last 3 commits
```

### Squashing Commits

You can squash multiple commits into one during an interactive rebase:

```
git rebase -i HEAD~3
# In the editor change "pick" to "squash" or "s" for commits you want to squash
```

### Resolving Merge Conflicts

When you get merge conflicts during a rebase:

```
# Fix conflicts in the files
git add <resolved-files>
git rebase --continue
```

## GitHub Issues and Pull Requests

### Referencing Issues

In commit messages or PR descriptions, use #issue-number to reference issues:

```
Fix the bug causing errors in tutorial notebooks

Closes #123
```

### PR Checklist

Before submitting a PR, ensure:

1. All tests pass locally
2. Documentation is updated if needed
3. Your code follows the project's style guide
4. Your branch is rebased on the latest master

## Advanced Git

### Git Bisect

Find the commit that introduced a bug:

```
git bisect start
git bisect bad  # Current commit is bad
git bisect good v1.0.0  # Last known good commit
# Git will checkout commits for you to test
git bisect good  # Current commit is good
git bisect bad   # Current commit is bad
# Continue until git identifies the first bad commit
git bisect reset  # When done
```

### Git Hooks

Useful for automatically running tests or linting before commits:

```
# In .git/hooks/pre-commit (make executable with chmod +x)
#!/bin/sh
pytest -xvs tests/
```

## Troubleshooting

### Undo Last Commit (not pushed)

```
git reset --soft HEAD~1
```

### Undo Last Push

```
git push -f origin HEAD^:master
```

### Recovering Lost Commits

```
git reflog
git checkout -b recovery-branch <commit-hash>
```

## Git Resources

- [Pro Git Book](https://git-scm.com/book/en/v2)
- [GitHub Guides](https://guides.github.com/)
- [Oh Shit, Git!?!](https://ohshitgit.com/) - For when things go wrong 