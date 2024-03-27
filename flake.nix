{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.nixpkgs-typst.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
    nixpkgs-typst,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
    typst-pkgs = import nixpkgs-typst {
        inherit system;
    };
    matplotlib = pkgs.python3Packages.matplotlib.override {enableQt = true;};
    pyEnv = pkgs.python3.withPackages (ps: with ps; [torchvision-bin onnx numpy pillow jupyter matplotlib ipython scipy pyqt5 tensorboard]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      default = pkgs.mkShell {
        packages = [pyEnv pkgs.jq typst-pkgs.typst];
        QT_PLUGIN_PATH = with pkgs.qt5; "${qtbase}/${qtbase.qtPluginPrefix}";
      };
    };
  };
}
