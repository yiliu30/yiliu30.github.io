+++
title = 'Getting Started with Hugo: A Simple Guide'
date = 2026-01-25T10:00:00+00:00
draft = false
tags = ['hugo', 'web development', 'tutorial']
categories = ['Programming']
+++

Welcome to my first blog post! In this article, I'll share a quick guide on building a personal website using Hugo, a fast and flexible static site generator.

<!--more-->

## Why Hugo?

Hugo is an excellent choice for building personal websites and blogs because:

- **Speed**: Hugo builds sites incredibly fast, even with thousands of pages
- **Simplicity**: Write content in Markdown and let Hugo handle the rest
- **Flexibility**: Customize your site with themes and templates
- **No Database Required**: Everything is file-based, making it easy to version control

## Quick Start

### Installation

First, install Hugo on your system:

```bash
# macOS
brew install hugo

# Linux (Debian/Ubuntu)
sudo apt install hugo

# Windows (using Chocolatey)
choco install hugo-extended
```

### Create a New Site

```bash
# Create a new Hugo site
hugo new site my-blog
cd my-blog

# Initialize git repository
git init
```

### Add a Theme

Hugo has many beautiful themes available. Let's use PaperMod as an example:

```bash
# Add PaperMod theme as a git submodule
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

### Create Your First Post

```bash
# Create a new blog post
hugo new posts/my-first-post.md
```

Edit the post in `content/posts/my-first-post.md`:

```markdown
+++
title = 'My First Post'
date = 2026-01-25T10:00:00+00:00
draft = false
+++

Hello World! This is my first blog post.
```

### Preview Your Site

```bash
# Start the Hugo development server
hugo server -D

# Visit http://localhost:1313 in your browser
```

## Customization Tips

Here are some ways to make your site unique:

1. **Custom CSS**: Add your styles in `assets/css/extended/custom.css`
2. **Configuration**: Edit `hugo.toml` to set your site title, description, and social links
3. **About Page**: Create `content/about.md` to tell your story
4. **Syntax Highlighting**: Hugo supports code highlighting out of the box

## Deployment

You can deploy your Hugo site to various platforms:

- **GitHub Pages**: Free hosting with GitHub Actions
- **Netlify**: Automatic builds and deployments
- **Vercel**: Fast global CDN deployment
- **AWS S3**: Host static files on AWS

## Example Code

Here's a simple Python function to demonstrate syntax highlighting:

```python
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    a, b = 0, 1
    result = []
    
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    
    return result

# Generate first 10 Fibonacci numbers
print(fibonacci(10))
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## Useful Resources

- [Hugo Documentation](https://gohugo.io/documentation/)
- [Hugo Themes](https://themes.gohugo.io/)
- [Markdown Guide](https://www.markdownguide.org/)

## Conclusion

Hugo makes it easy to create a beautiful, fast website without dealing with complex setups. Whether you're starting a personal blog, portfolio, or documentation site, Hugo provides the tools you need to get started quickly.

Happy blogging! 🚀
