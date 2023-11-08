{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgsgpu = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; config.cudaSupport=true; };
    pkgscpu = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; config.cudaSupport=false; };
    pkgs = import nixpkgs { system = "x86_64-linux"; };
    # TODO: Do I need two separate environments, or will CPU still work with cuda version of torch?
    pyEnvGPU = pkgsgpu.python3.withPackages(ps: with ps; [ torch torchvision onnx numpy pillow ]);
    pyEnvCPU = pkgscpu.python3.withPackages(ps: with ps; [ torch torchvision onnx numpy pillow matplotlib jupyter ipython ]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      gpu = pkgs.mkShell {
        packages = [ pyEnvGPU ];
      };
      cpu = pkgs.mkShell {
        packages = [ pyEnvCPU ];
        shellHook = "jupyter notebook";
      };
    };
  };
}
