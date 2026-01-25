# My Personal Blog

A personal website built with [Hugo](https://gohugo.io/), a fast and flexible static site generator.

## Features

- 📝 Blog posts written in Markdown
- 🎨 Clean and responsive design with PaperMod theme
- 🚀 Fast page loads
- 📱 Mobile-friendly
- 🔍 SEO optimized
- 💬 Syntax highlighting for code blocks
- 🌙 Dark/Light mode support

## Prerequisites

- [Hugo Extended](https://gohugo.io/installation/) (v0.120.0 or later)
- [Git](https://git-scm.com/)

## Local Development

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Add the PaperMod theme**

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive
```

3. **Start the development server**

```bash
hugo server -D
```

4. **View your site**

Open your browser and navigate to `http://localhost:1313`

## Creating New Posts

Create a new blog post with:

```bash
hugo new posts/my-new-post.md
```

Edit the file in `content/posts/my-new-post.md` and set `draft: false` when ready to publish.

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── hugo.yml         # GitHub Actions for deployment
├── assets/
│   └── css/
│       └── extended/
│           └── custom.css   # Custom styles
├── content/
│   └── posts/               # Blog posts go here
├── static/                  # Static files (images, etc.)
├── themes/
│   └── PaperMod/           # Theme submodule
├── hugo.toml               # Hugo configuration
└── README.md
```

## Configuration

Edit `hugo.toml` to customize:

- Site title and description
- Social media links
- Menu items
- Theme parameters

## Deployment

### GitHub Pages

This site is configured to automatically deploy to GitHub Pages using GitHub Actions.

**Setup:**

1. Go to your repository settings
2. Navigate to Pages section
3. Under "Build and deployment", select "GitHub Actions" as the source
4. Push to the `main` branch to trigger deployment

Your site will be available at `https://yourusername.github.io/`

### Other Hosting Options

- **Netlify**: Connect your GitHub repo for automatic deployments
- **Vercel**: Import your GitHub repo for instant deployment
- **AWS S3**: Build with `hugo` and upload the `public/` folder

## Writing Tips

### Front Matter

Each post should have front matter at the top:

```yaml
+++
title = 'Your Post Title'
date = 2026-01-25T10:00:00+00:00
draft = false
tags = ['tag1', 'tag2']
categories = ['Category']
+++
```

### Markdown Basics

- Use `#` for headings (# H1, ## H2, ### H3, etc.)
- Use `**bold**` for **bold text**
- Use `*italic*` for *italic text*
- Use `` `code` `` for inline code
- Use triple backticks for code blocks with syntax highlighting

### Code Blocks

````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

## Customization

### Custom CSS

Add custom styles to `assets/css/extended/custom.css`

### Adding Pages

Create new pages:

```bash
hugo new about.md
hugo new contact.md
```

## Resources

- [Hugo Documentation](https://gohugo.io/documentation/)
- [PaperMod Theme Documentation](https://github.com/adityatelange/hugo-PaperMod)
- [Markdown Guide](https://www.markdownguide.org/)
- [Hugo Themes](https://themes.gohugo.io/)

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

Feel free to reach out:

- GitHub: [@yourusername](https://github.com/yourusername)
- Twitter: [@yourusername](https://twitter.com/yourusername)
- Email: your.email@example.com

---

Built with ❤️ using [Hugo](https://gohugo.io/) and [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
