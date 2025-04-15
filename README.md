## Ghost Vim plugin

### Status

This is mostly developed as a hobby project and intended for personal use.
It is very unstable and not recommended for production use.

It is in fact not recommended for any use at all. It may eat your computer or 
try and take over mankind. It may also cause your computer to be permanently haunted.
Use at your own risk. I am not responsible for any damage caused by this plugin.

### Introduction

#### A pipeline, agent-based C code assistant for Vim running on locall ollama

Ghost is a plugin for Vim that offers a pipeline of agents to perform various tasks
within an existing codebase. Starting from a prompt command, the agents
analyze the codebase, the documentation and complete their knowledge of the task using
a web search tool.

Eventually, a diff is generated with the changes proposed by the agents pipeline, which
can be reviewed and applied to the codebase, or rejected.

### Installation

To install Ghost.vim, you can use your favorite Vim plugin manager. For example,
I use Vundle, so I add the following line to my `.vimrc` in vundle's plugins section:

```vim
Plugin danielinux/ghost.vim
```

Then run `:VundleInstall` in Vim.

#### Usage

Once installed, you can start the pipeline by running the command `:GhostRun`. This will prompt you to enter a task description. The agents will then analyze the codebase and generate a diff with the proposed changes.

You can review the diff, navigating between files via `:GhostPrev`/`:GhostNext`, apply it to your codebase using `:GhostAccept`, or discard using `:GhostReject`.


#### Configuration

Nothing can be configured at this time. More configuration options will be added in future releases.

### License

This software is released under AGPL-3.0 license. Please see COPYING file for more information.





