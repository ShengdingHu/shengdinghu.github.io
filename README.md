
```bash
hugo new site myhomepage --format yaml
cd myhomepage
git init
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
echo "theme: 'PaperMod'" >> hugo.yaml
hugo server
```
