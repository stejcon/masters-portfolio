{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  nixConfig = {
    allowUnfree = true;
    cudaSupport = true;
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system}.pkgs;
    pyEnv = pkgs.python3.withPackages(ps: with ps; [ torch.override{cudaSupport=true;} torchvision onnx numpy pillow ]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      default = pkgs.mkShell {
        packages = [ pyEnv ];
      };
    };
  };
}
