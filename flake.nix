{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.cudaSupport=true; };
    pyEnv = pkgs.python3.withPackages(ps: with ps; [ torchvision-bin onnx numpy pillow matplotlib jupyter ipython ]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      default = pkgs.mkShell {
        packages = [ pyEnv pkgs.typst ];
      };
    };
  };
}
