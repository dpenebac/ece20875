#!/usr/bin/python3

# In the following lines, choose the correct git commands to perform each of the actions.
# Print the corresponding variable following the print example given.

# We have defined a variable called 'filename' that holds the file, 'git_commands.py' that you need to work with
filename = 'git_commands.py'

# Commands
cmd0 = 'Replace This'  # Relace cmd0 in each question with the correct command (e.g, cmd1)
cmd = 'git'
cmd1 = 'pull'
cmd2 = 'add --all'
cmd3 = 'clone'
cmd4 = 'commit -m'
cmd5 = 'add'
cmd6 = 'push'

# Example #
print('Example: What command should I use to download my assignment from GitHub?')
print(cmd + ' ' + cmd1 + ' https://github.com/johndoe/ConfusedStudent.git')

# Question 1
print('Question 1: If `git_commands.py` is a new working file that I`ve just created on my local machine, then what commands should I use to submit it to the remote repository?')
# Step 1: Stage the file for the next version
# Step 2: Create a new version (update local repository)
# Step 3: Update the remote repository
print('step 1 ', cmd + ' ' + cmd5 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 2 ', cmd + ' ' + cmd4 + ' ' + "'First version of file is done. I should pay my consultant.'")  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 3 ', cmd + ' ' + cmd6 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)

# Question 2
print('Question 2: If I modified `git_commands.py` on my local machine, then what commands should I use to update the file on the remote repository?')
# Step 1: Stage the file for the next version
# Step 2: Create a new version (update local repository)
# Step 3: Update the remote repository
print('step 1 ', cmd + ' ' + cmd5 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 2 ', cmd + ' ' + cmd4 + ' ' + "'Problem fixed and file updated. My consultant is cool and deserves $20.'")  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 3 ', cmd + ' ' + cmd6 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)

# Question 3
print('Question 3: If `git_commands.py` has been removed from my local machine, then what commands should I use to remove it from the remote repository?')
# Step 1: Stage the removal of the file
# Step 2: Create a new version
# Step 3: Update the repository
print('step 1 ', cmd + ' ' + cmd5 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 2 ', cmd + ' ' + cmd4 + ' ' + "'Removed File from Repository.'")  # Replace cmd0 in each question with the correct command (e.g, cmd1)
print('step 3 ', cmd + ' ' + cmd6 + ' ' + filename)  # Replace cmd0 in each question with the correct command (e.g, cmd1)

# Question 4
print('Question 4: If the remote repository on Github has new changes that are not on my local machine, what command should I use to update my local machine?')
print(cmd + ' ' + cmd1)  # Replace cmd0 in each question with the correct command (e.g, cmd1)

