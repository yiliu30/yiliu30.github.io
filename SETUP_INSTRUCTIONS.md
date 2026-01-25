# Setup Instructions

## Giscus Comments (Already Configured!)
✅ GitHub Discussions enabled
✅ Configuration added to hugo.toml

**Final Step:** Install the Giscus app:
1. Visit: https://github.com/apps/giscus
2. Click "Install" 
3. Select your repository: yiliu30/yiliu30.github.io
4. Click "Install"

That's it! Comments with reactions (like stars ⭐) will appear on all blog posts.

## GoatCounter Analytics

**Setup Steps:**
1. Visit: https://www.goatcounter.com/signup
2. Create a free account with code: "yiliu30" (or choose your own)
3. Your analytics will be available at: https://yiliu30.goatcounter.com
4. If you chose a different code, update hugo.toml:
   ```toml
   [params.analytics.goatcounter]
     code = "your-code-here"
   ```

**Features you'll get:**
- Page view counts on each post
- Privacy-friendly (no cookies, GDPR compliant)
- Real-time analytics dashboard
- Free forever for reasonable traffic

## Testing
After deploying, visit your blog and you should see:
- Comments section at the bottom of each post
- Visitors can leave comments and react with emojis
- View counts will start accumulating (visible in your GoatCounter dashboard)

