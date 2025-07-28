# 🔐 API Key Setup Guide

This guide shows you how to securely set up your Anthropic API key for the D&D simulation project.

## 🚀 Quick Setup (Recommended)

### Step 1: Get Your API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (it starts with `sk-ant-`)

### Step 2: Set Up the Key File
1. Open the `api_key.txt` file in the project root directory
2. Replace the placeholder text with your actual API key
3. Save the file
4. **Done!** The file is already in `.gitignore` so it won't be committed

### Step 3: Test Your Setup
1. Open `tutorials/llm_gameplay_tutorial.ipynb`
2. Run the API key setup cell
3. You should see: `✅ API key loaded from api_key.txt`

## 🔒 Security Features

### What's Protected
- ✅ `api_key.txt` is in `.gitignore` - never committed to repository
- ✅ Multiple API key patterns excluded from version control
- ✅ Secure file reading with error handling
- ✅ Key prefix display only (never shows full key)

### File Permissions (Optional)
Make the key file readable only by you:
```bash
chmod 600 api_key.txt
```

## 🔧 Alternative Setup Methods

### Option 1: Environment Variable
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Option 2: Direct Assignment (Not Recommended)
```python
api_key = 'your_key_here'  # Don't do this in production!
```

## 🚨 Troubleshooting

### Problem: "api_key.txt found but contains placeholder text"
**Solution:** Edit `api_key.txt` and replace `PASTE_YOUR_ANTHROPIC_API_KEY_HERE` with your actual key.

### Problem: "No API key found!"
**Solutions:**
1. Check that `api_key.txt` exists in the project root
2. Make sure your key is on the last line after the comments
3. Verify your key starts with `sk-ant-`
4. Try setting the environment variable instead

### Problem: "Error reading api_key.txt"
**Solutions:**
1. Check file permissions: `ls -la api_key.txt`
2. Make sure the file isn't corrupted
3. Try recreating the file with just your key

### Problem: API calls failing
**Solutions:**
1. Verify your API key is valid at [Anthropic Console](https://console.anthropic.com/)
2. Check your account has sufficient credits
3. Ensure you're using the correct key format

## 📁 File Structure

```
dnd-dynamics/
├── api_key.txt              # ← Your API key goes here
├── .gitignore               # ← Contains api_key.txt exclusion
├── tutorials/
│   └── llm_gameplay_tutorial.ipynb  # ← Uses secure key loading
└── llm_scaffolding/
    └── dnd_simulation.py    # ← Main simulation code
```

## 🛡️ What If I Accidentally Commit My Key?

If you accidentally commit your API key:

1. **Immediately revoke the key** at [Anthropic Console](https://console.anthropic.com/)
2. Generate a new API key
3. Update your `api_key.txt` file with the new key
4. Remove the key from Git history:
   ```bash
   git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch api_key.txt' --prune-empty --tag-name-filter cat -- --all
   ```
5. Force push to update remote repository:
   ```bash
   git push --force --all
   ```

## ✅ Best Practices

- ✅ Use the `api_key.txt` file method (recommended)
- ✅ Never share your API key or commit it to version control
- ✅ Regularly rotate your API keys
- ✅ Monitor your API usage at [Anthropic Console](https://console.anthropic.com/)
- ✅ Keep your key file permissions restrictive (`chmod 600`)

## 🎯 Ready to Go!

Once your API key is set up, you can:
1. Run the tutorial notebook: `tutorials/llm_gameplay_tutorial.ipynb`
2. Create your own D&D simulations
3. Experiment with different character configurations
4. Analyze AI vs human gameplay patterns

Happy adventuring! 🐉⚔️