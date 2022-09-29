{ pkgs ? import <nixpkgs> {}}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustc
    cargo
    rustfmt
    rust-analyzer
    clippy
    trunk
    clang
    libiconv
    gitMinimal
  ];

  RUST_BACKTRACE = 1;
}