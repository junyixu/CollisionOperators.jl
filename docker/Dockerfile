# syntax=docker/dockerfile:1
# GPU runtime image for Runpod pods: Julia + every dependency of main.jl
# (including CUDA.jl and its toolkit artifacts) installed and precompiled at
# build time, so a pod goes from boot to the first time step without Pkg work.
# On top of that, a root-only dev environment (neovim + plugins, zsh, yazi, …)
# for editing and inspecting runs on pods reachable only via the SSH proxy.
#
# All stages are archlinux:base and every tool comes from pacman, except Julia,
# which is juliaup-bin from the AUR. The base tag is not pinned: a rebuild that
# misses the layer cache picks up whatever the Arch repos ship that day.
#
# Build (context = repository root):
#   docker build -f docker/Dockerfile -t ghcr.io/junyixu/collisionoperators .
#
# Run inside the pod (JULIA_PROJECT already points at the baked environment —
# do NOT pass --project=., the repository root has no Manifest):
#   cd /root/CollisionOperators.jl && git pull
#   julia main.jl parameters_LB_sq_d04.jl --use_gpu=true ...

# ---- juliaup (AUR build stage) ----------------------------------------------
# makepkg refuses to run as root: build juliaup-bin as a throwaway user here and
# hand only the package file to the runtime stage, so the final image carries
# neither base-devel nor a sudo-capable build user. The glob skips the -debug
# split package that makepkg may emit.
FROM archlinux:base AS aur
RUN pacman -Syu --noconfirm --needed base-devel git \
 && useradd -m builder \
 && echo 'builder ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/builder
USER builder
RUN git clone --depth=1 https://aur.archlinux.org/juliaup-bin.git ~/juliaup-bin \
 && cd ~/juliaup-bin \
 && makepkg -sir --noconfirm \
 && mkdir ~/pkg && cp juliaup-bin-[0-9]*.pkg.tar.zst ~/pkg/

# ---- dev tools (build stage) -------------------------------------------------
# Same distro as the runtime stage, so the compiled treesitter parsers and
# plugin libraries run unchanged there. Built as root (HOME=/root): the configs
# and plugin caches are copied to the same paths in the final stage. nodejs is
# for `tree-sitter generate` (the latex grammar ships no parser.c); eza / atuin /
# yazi are on PATH for the first zsh start below, as in the runtime image.
FROM archlinux:base AS devtools
ARG PARSERS="bash c diff html lua luadoc markdown markdown_inline query vim vimdoc python yaml latex typst"
ENV HOME=/root PATH=/root/.local/bin:$PATH
RUN pacman -Syu --noconfirm --needed \
      ca-certificates curl git gcc make nodejs unzip zsh neovim tree-sitter-cli eza atuin yazi \
 && rm -rf /var/cache/pacman/pkg/*
# Configs come straight from public GitHub repos. The *_REV args are cache
# busters only: the Docker workflow passes each branch's current commit, so a
# config push invalidates its layer (and everything after it in this stage).
ARG NVIM_BRANCH=main
ARG NVIM_REV=unset
RUN echo "nvim-config ${NVIM_BRANCH} @ ${NVIM_REV}" \
 && git clone --depth=1 --branch "${NVIM_BRANCH}" https://github.com/junyixu/nvim-config.git ~/.config/nvim \
 && git -C ~/.config/nvim rev-parse HEAD > ~/.config/nvim-rev
ARG ZSH_BRANCH=main
ARG ZSH_REV=unset
RUN echo "zsh-config ${ZSH_BRANCH} @ ${ZSH_REV}" \
 && git clone --depth=1 --branch "${ZSH_BRANCH}" https://github.com/junyixu/zsh-config-public.git ~/.config/zsh \
 && git -C ~/.config/zsh rev-parse HEAD > ~/.config/zsh-rev \
 && rm -rf ~/.config/zsh/.git
ARG YAZI_BRANCH=main
ARG YAZI_REV=unset
RUN echo "yazi-config ${YAZI_BRANCH} @ ${YAZI_REV}" \
 && git clone --depth=1 --branch "${YAZI_BRANCH}" https://github.com/junyixu/yazi-config-public.git ~/.config/yazi \
 && git -C ~/.config/yazi rev-parse HEAD > ~/.config/yazi-rev \
 && rm -rf ~/.config/yazi/.git
RUN nvim --headless "+Lazy! install" +qa \
 && nvim --headless -c "lua require('nvim-treesitter').install(vim.split('${PARSERS}', ' ')):wait(900000)" -c qa
# blink.cmp fetches its prebuilt fuzzy lib on first load. The download writes `version`=v0.0.0 first and the real
# tag only after the checksum passes, so wait for that (not just for the .so) before nvim exits.
RUN for i in 1 2 3; do \
      nvim --headless \
        -c "lua require('lazy').load({ plugins = { 'blink.cmp' } })" \
        -c "lua local f = vim.fn.stdpath('data') .. '/lazy/blink.cmp/target/release/version'; print(vim.wait(120000, function() local h = io.open(f); if not h then return false end; local v = h:read('*a'); h:close(); return v ~= '' and v ~= 'v0.0.0' end, 200))" \
        -c "lua vim.wait(1000)" \
        -c qa 2>&1; \
      v=$(cat ~/.local/share/nvim/lazy/blink.cmp/target/release/version 2>/dev/null); \
      [ -n "$v" ] && [ "$v" != v0.0.0 ] && break; \
      echo "blink fuzzy download attempt $i failed" >&2; \
    done; \
    v=$(cat ~/.local/share/nvim/lazy/blink.cmp/target/release/version); echo "blink fuzzy lib version: $v"; \
    [ "$v" != v0.0.0 ] && test -f ~/.local/share/nvim/lazy/blink.cmp/target/release/libblink_cmp_fuzzy.so
RUN echo 'for f in ~/.config/zsh/zshrc.d/*.zsh(N); do source "$f"; done && typeset -U path PATH' > ~/.zshrc
# first interactive start clones zinit + every plugin/snippet, so bake it here
RUN TERM=xterm-256color zsh -ic 'exit' </dev/null
# p10k downloads gitstatusd on the first prompt; fetch it at build time instead
RUN ~/.local/share/zinit/plugins/romkatv---powerlevel10k/gitstatus/install
# slim: drop git history and per-plugin test dirs (no :Lazy update / zinit update inside the image; rebuild to update)
# blink.cmp keeps its .git: it runs `git describe --tags` to pick the matching prebuilt fuzzy lib
RUN find ~/.local/share/nvim/lazy ~/.local/share/zinit -path '*/lazy/blink.cmp/.git' -prune -o -name .git -prune -exec rm -rf {} + \
 && find ~/.local/share/nvim/lazy ~/.local/share/zinit/plugins -mindepth 2 -maxdepth 2 -type d \( -name test -o -name tests \) -prune -exec rm -rf {} +

# ---- runtime image -------------------------------------------------------------
FROM archlinux:base

ARG JULIA_VERSION=1.12.7

# Precompiled package images are CPU-specific. Build for the same multi-target
# set as the official Julia binaries, so the images stay valid on whatever host
# CPU the pod lands on instead of being recompiled at first `using`.
ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PROJECT=/opt/collisionops-env \
    JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"

# juliaup keeps its own state in ~/.julia/juliaup (it ignores JULIA_DEPOT_PATH).
# Exactly one Julia in the image: no version picked from manifests, no
# versions-db refresh in the background of every `julia` start.
COPY --from=aur /home/builder/pkg/ /tmp/pkg/
RUN pacman -Syu --noconfirm \
 && pacman -U --noconfirm /tmp/pkg/*.pkg.tar.zst \
 && rm -rf /tmp/pkg /var/cache/pacman/pkg/* \
 && juliaup add "${JULIA_VERSION}" \
 && juliaup default "${JULIA_VERSION}" \
 && juliaup config manifestversiondetect false \
 && juliaup config versionsdbupdateinterval 0 \
 && julia --version

# Dependencies first, so a code-only change reuses this (slow) layer.
COPY docker/Project.toml docker/Manifest.toml docker/LocalPreferences.toml $JULIA_PROJECT/
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(; strict = true)' \
    && julia -e 'using CUDA, CairoMakie, Mantis' \
    && find $JULIA_DEPOT_PATH/artifacts -name 'libcudart.so.*' | grep -q . \
    && chmod -R a+rX $JULIA_DEPOT_PATH

# Everything below sits after the Julia depot layer, so changing the dev tools
# or the shell environment never rebuilds the depot.

# Pods expose fewer vCPUs than the host has cores, but OpenBLAS sizes its pool
# by host cores; the oversubscribed threads spin and starve the GPU. One thread
# was fastest for the Anderson least-squares on a 12-vCPU RTX 4090 pod.
ENV OPENBLAS_NUM_THREADS=1

# Dev tools and pod services: packages here, nvim plugins / zsh plugins /
# configs from the devtools stage. Root only. python3 is for the cuInit smoke
# test run right after a pod boots.
RUN pacman -Syu --noconfirm --needed \
      openssh git zsh tmux fzf zoxide direnv bat ripgrep fd file less kitty-terminfo \
      ctags global htop eza atuin yazi neovim rclone python \
 && rm -rf /var/cache/pacman/pkg/* \
 && usermod -s /usr/bin/zsh root \
 && echo 'LANG=C.UTF-8' > /etc/locale.conf
COPY --from=devtools /root/.config /root/.config
COPY --from=devtools /root/.local /root/.local
COPY --from=devtools /root/.zshrc /root/.zshrc
COPY --from=devtools /root/.cache /root/.cache

# Image ENV reaches only PID 1, not SSH sessions: replay it for login shells
# (/etc/profile.d) and for every zsh, including `ssh pod 'cmd'` (/etc/zsh/zshenv;
# zsh reads /etc/profile only as a login shell).
RUN { echo "export JULIA_DEPOT_PATH='$JULIA_DEPOT_PATH'"; \
      echo "export JULIA_PROJECT='$JULIA_PROJECT'"; \
      echo "export JULIA_CPU_TARGET='$JULIA_CPU_TARGET'"; \
      echo "export OPENBLAS_NUM_THREADS='$OPENBLAS_NUM_THREADS'"; \
    } > /etc/profile.d/julia.sh \
 && echo '[ -r /etc/profile.d/julia.sh ] && emulate sh -c ". /etc/profile.d/julia.sh"' >> /etc/zsh/zshenv

# Runpod's SSH proxy ignores root's login shell: it starts bash with TERM=xterm,
# no locale and a high-latency tty (nvim's startup terminal queries time out
# with E1568). Fix those up and switch to zsh. A direct SSH session goes through
# sshd with the client's TERM and lands in zsh, so it never hits this.
# /start.sh appends `source /etc/rp_environment` to .bashrc at boot, after this
# block, so load it here before exec'ing zsh. Arch's /root has no dotfiles, so
# also add the .bash_profile that makes a login bash read .bashrc.
RUN echo '[[ -f ~/.bashrc ]] && . ~/.bashrc' > /root/.bash_profile && cat >> /root/.bashrc <<'EOF'

# Runpod SSH proxy session → 256-color TERM, UTF-8, nvim without tty queries, zsh
if [[ $- == *i* && $TERM == xterm ]]; then
    [ -r /etc/rp_environment ] && . /etc/rp_environment
    export TERM=xterm-256color COLORTERM=truecolor LANG=C.UTF-8 NVIM_NOTTYFAST=1
    exec zsh -l
fi
EOF

# Pod entrypoint in place of runpod/base's: sshd from PUBLIC_KEY, pod env for
# SSH sessions, then idle.
COPY --chmod=755 docker/start.sh /start.sh
CMD ["/start.sh"]

COPY . /root/CollisionOperators.jl
WORKDIR /root/CollisionOperators.jl
