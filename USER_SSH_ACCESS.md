# Access `rabelais` with SSH

First you'll need to ask an administrator to create you an account on the `rabelais` workstation.
You can do this by posting in the [#eng-rabelais](https://valencelabsworkspace.slack.com/archives/C05AZFX2T26) channel on Slack, 
and providing the public key of a new SSH key pair. An administrator will get back to you with a username, and a password.

## Access the cluster

In `$HOME/.ssh/config` add:

```
Host rabelais
    HostName 9.tcp.ngrok.io
    Port 23615
    User <YOUR_USERNAME>
    IdentityFile <PATH/TO/RABELAIS/PRIVATE/KEY>
```

Then simply do `ssh rabelais`.
