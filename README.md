<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>


**Pie** is a high-performance, programmable LLM serving system that empowers you to design and deploy custom inference logic and optimization strategies.

> **Note** 🧪
>
> This software is in a **pre-release** stage and under active development. It's recommended for testing and research purposes only. For the research artifact associated with our [SOSP paper](https://ingim.org/papers/gim2025pie.pdf), please refer to the `sosp25` branch.


## Getting started


```bash
pip install pie-server
```

Quick start:

```bash
pie config init
pie model download Qwen/Qwen3-0.6B

pie run text-completion -- --prompt "Hello world"
```


Check out the [https://pie-project.org/docs](https://pie-project.org/) for more information.

## Community

**Issues & Bugs**: Please report bugs on [GitHub Issues](https://github.com/pie-project/pie/issues).

**Discussions**: Have a question or feedback? Join us on [GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE)