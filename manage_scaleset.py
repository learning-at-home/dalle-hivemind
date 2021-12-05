import os
from argparse import ArgumentParser
from base64 import b64encode

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient


print("=======================WARNING=======================")
print("= The code may fail to import 'gi' but that is okay =")
print("===================END OF WARNING====================")
SUBSCRIPTION_ID = os.environ["SUBSCRIPTION_ID"]
GROUP_NAME = "dalle_west2"
NETWORK_NAME = "vnet"
SUBNET_NAME = "subnet"
LOCATION = "westus2"
ADMIN_PASS = os.environ['AZURE_PASS']

SCALE_SETS = ('worker',)
SWARM_SIZE = 4

WORKER_CLOUD_INIT = """#cloud-config
package_update: true
packages:
  - build-essential
  - wget
  - git
  - vim
write_files:
  - path: /home/hivemind/init_worker.sh
    permissions: '0766'
    owner: root:root
    content: |
      #!/usr/bin/env bash
      set -e
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
      bash install_miniconda.sh -b -p /opt/conda
      export PATH="/opt/conda/bin:${PATH}"
      conda install python~=3.8.0 pip
      conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia
      conda clean --all
      pip install https://github.com/learning-at-home/hivemind/archive/scaling_tweaks.zip
      systemctl enable testserv
      systemctl start testserv
  - path: /etc/systemd/system/testserv.service
    permissions: '0777'
    owner: root:root
    content: |
      [Unit]
      Description=One Shot

      [Service]
      ExecStart=/etc/createfile
      Type=oneshot
      RemainAfterExit=yes

      [Install]
      WantedBy=multi-user.target
  - path: /etc/createfile
    permissions: '0777'
    owner: root:root
    content: |
      #!/bin/bash
      export PATH="/opt/conda/bin:${PATH}"
      cd /home/hivemind
      ulimit -n 8192
      
      git clone https://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@github.com/learning-at-home/dalle-hivemind.git -b azure
      cd dalle-hivemind
      pip install -r requirements.txt
      pip install -U transformers==4.10.2 datasets==1.11.0
      
      WANDB_API_KEY=7cc938e45e63ef7d2f88f811be240ba0395c02dd python run_trainer.py --run_name $(hostname) \
         --experiment_prefix dalle_large_5groups \
         --initial_peers /ip4/52.232.13.142/tcp/31334/p2p/QmZLrSPKAcP4puJ8gUGvQ155thk5Q6J7oE5exMUSq1oD5i \
         --per_device_train_batch_size 1 --gradient_accumulation_steps 1
runcmd:
  - bash /home/hivemind/init_worker.sh
"""


def main():
    parser = ArgumentParser()
    parser.add_argument('command', choices=('create', 'delete'))
    args = parser.parse_args()

    resource_client = ResourceManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID
    )
    network_client = NetworkManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID
    )
    compute_client = ComputeManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID
    )

    # Create resource group
    resource_client.resource_groups.create_or_update(
        GROUP_NAME,
        {"location": LOCATION}
    )

    # Create virtual network
    network_client.virtual_networks.begin_create_or_update(
        GROUP_NAME,
        NETWORK_NAME,
        {
            'location': LOCATION,
            'address_space': {
                'address_prefixes': ['10.0.0.0/16']
            }
        }
    ).result()

    subnet = network_client.subnets.begin_create_or_update(
        GROUP_NAME,
        NETWORK_NAME,
        SUBNET_NAME,
        {'address_prefix': '10.0.0.0/16'}
    ).result()

    if args.command == 'create':

        scalesets = []

        for scaleset_name in SCALE_SETS:
            cloud_init_cmd = WORKER_CLOUD_INIT
            vm_image = {
                "exactVersion": "21.06.0",
                "offer": "ngc_base_image_version_b",
                "publisher": "nvidia",
                "sku": "gen2_21-06-0",
                "version": "latest",
            }

            vm_config = {
                "sku": {
                    "tier": "Standard",
                    "capacity": SWARM_SIZE,
                    "name": "Standard_NC4as_T4_v3"
                },
                "plan": {
                    "name": "gen2_21-06-0",
                    "publisher": "nvidia",
                    "product": "ngc_base_image_version_b"
                },
                "location": LOCATION,
                "virtual_machine_profile": {
                    "storage_profile": {
                        "image_reference": vm_image,
                        "os_disk": {
                            "caching": "ReadWrite",
                            "managed_disk": {"storage_account_type": "Standard_LRS"},
                            "create_option": "FromImage",
                            "disk_size_gb": "32",
                        },
                    },
                    "os_profile": {
                        "computer_name_prefix": scaleset_name,
                        "admin_username": "hivemind",
                        "admin_password": ADMIN_PASS,
                        "linux_configuration": {
                            "disable_password_authentication": True,
                            "ssh": {
                                "public_keys": [
                                    {
                                        "key_data": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDPFugAsrqEsqxj+hKDTfgrtkY26jqCjRubT5vhnJLhtkDAqe5vJ1donWfUVhtBfnqGr92LPmJewPUd9hRa1i33FLVVdkFAs5/Cg8/YbzR8B8e1Y+Nl5HeT7Dq1i+cPEbA1EZAm9tqK4VWYeCMd3CDkoJVuweTwyja08mxtnVNwKCeY4oBKQCE5QlliAKaQnGpJE6MRnbudWM9Ly1wM6OaJVdGwsfPfEG/sSDip4q/8x/KGAzKbhE6ax15Yu/Bu12ahcIdScQsYK9Y6Sm57MHQQLWQO1G+3h3oCTXQ0BGaSMWKXsjmHsB7f9kLZ1j8yMoGlgbpWbjB0ZVsK/4Zh8Ho3h9gDXADzt1j69qT1aERWCt7fxp9+WOLsCTw1W/W9FY2Ia4niVh2/wEwT9AcOBcAqBl7kXQAoUpP8b2Xb+KNXyTEtVB562EdFn+LmG1gZAy8J3piy2/zoo16QJP5PjpKW5GFxL6BRYLtG+uxgx1Glya617T0dtJF/X2vxjT45QK3FaFH1Zd+vhpcLg94fOPNPEhNU7EeBVp8CGYNd+aXVIPsb0I7EIVu9wWi3/a7y86cUedal61fEigfmAQkC7AHYiAiiT94eARj0N+KgjEy2UOITSCJJTHuamYWO8jZc/n7yAqr6mxOKn5ZjBTfAR9bNB/D+HpL6yepI1UDGBVk4DQ== justHeuristic@gmail.com\n",
                                        "path": "/home/hivemind/.ssh/authorized_keys"
                                    }
                                ]
                            }
                        },
                        "custom_data": b64encode(cloud_init_cmd.encode('utf-8')).decode('latin-1'),
                    },
                    "network_profile": {
                        "network_interface_configurations": [
                            {
                                "name": "test",
                                "primary": True,
                                "enable_accelerated_networking": True,
                                "ip_configurations": [
                                    {
                                        "name": "test",
                                        "subnet": {
                                            "id": f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{GROUP_NAME}/providers/Microsoft.Network/virtualNetworks/{NETWORK_NAME}/subnets/{SUBNET_NAME}"
                                        },
                                        "public_ip_address_configuration": {
                                            "name": "pub1",
                                            "idle_timeout_in_minutes": 15
                                        }

                                    }
                                ]
                            }
                        ]
                    },
                    "diagnostics_profile": {"boot_diagnostics": {"enabled": True}},
                    "priority": "spot",
                    "eviction_policy": "deallocate",
                },
                "upgrade_policy": {
                    "mode": "Manual"
                },
                "upgrade_mode": "Manual",
                "spot_restore_policy": {"enabled": True}
            }

            # Create virtual machine scale set
            vmss = compute_client.virtual_machine_scale_sets.begin_create_or_update(
                GROUP_NAME,
                scaleset_name,
                vm_config,
            )
            print(f"{scaleset_name} {vmss.status()}")
            scalesets.append(vmss)

        for scaleset_name, vmss in zip(SCALE_SETS, scalesets):
            print(f"Created scale set {scaleset_name}:\n{vmss.result()}")

    else:
        delete_results = []
        for scaleset_name in SCALE_SETS:
            delete_results.append(compute_client.virtual_machine_scale_sets.begin_delete(GROUP_NAME, scaleset_name))

        for scaleset_name, delete_result in zip(SCALE_SETS, delete_results):
            delete_result.result()
            print(f"Deleted scale set {scaleset_name}")


if __name__ == "__main__":
    main()
