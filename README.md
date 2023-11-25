<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD3 License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
<h2 align="center">empyrean</h2>
<p></p>

  <p align="center">
    N-Body Simulator with CUDA compute and OpenGL visualization support
    <br />
    <!-- <a href="https://github.com/dhmnr/empyrean"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <a href="https://github.com/dhmnr/empyrean/issues">Report Bug</a>
    ·
    <a href="https://github.com/dhmnr/empyrean/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#building">Building</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The Gravitational N-Body Simulator is a computational physics project designed to simulate the complex interactions between multiple celestial bodies under the influence of gravity. This project aims to provide a versatile tool for scientists, researchers, and astronomy enthusiasts to better understand the dynamics of celestial systems, from simple two-body interactions to intricate multi-body systems like galaxies and star clusters.<p align="right">(<a href="#top">back to top</a>)</p>



### Features

* Simulation is decoupled from visualization, allowing for extremely small integration steps. Which provids realistic N-Body simulation
* Exploiting the embarrasingly parallel nature of the N-body problem, Most of the work is offloaded to GPU by using CUDA kernels for mathematical calculations 
* CUDA kernels directly write to OpenGL vertex buffer objects in GPU memory, avoding extremely costly Host to Device and Device to Host data copies
* Provide a simple YAML initializer support for configuring initial state

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->



<!-- ROADMAP -->
## Roadmap

### Simulation Core
The core of empyrean, handles the physics calculations and simulation loop.
 - [ ]  Add support for more integretors (Verlet, Runge-Kutta IV, etc)
 - [ ]  Optimize reduction in CUDA kernels ([Ref](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf))
 - [ ]  Add [Barns-Hut](http://arborjs.org/docs/barnes-hut) approximation algorithm 
 - [ ]  Implement SI prefixes and scale units automatically


### Rendering and Visualization
The rendering component handles the visualization of the particles and the simulation.
 - [ ] Render anti-aliased circular points
 - [ ] Add more camera controls
 - [ ] Support pause/resume and controlling simulation speed 
 - [ ] Support Vulkan
 - [ ] Display FPS and other details on window
 - [ ] Provide a user interface widget before rendering to control simulation parameters
 - [ ] Motion blur for fast moving points


### Common
 - [ ] Add documentation

<!-- See the [open issues](https://github.com/dhmnr/empyrean/issues) for a full list of proposed features (and known issues). -->

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Dheemanth Manur - dheemanthmanur72@gmail.com

<!-- [@twitter_handle](https://twitter.com/twitter_handle) -->

Project Link: [https://github.com/dhmnr/empyrean](https://github.com/dhmnr/empyrean)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Choose an Open Source License](https://choosealicense.com)
* [Img Shields](https://shields.io)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dhmnr/empyrean.svg?style=for-the-badge
[contributors-url]: https://github.com/dhmnr/empyrean/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dhmnr/empyrean.svg?style=for-the-badge
[forks-url]: https://github.com/dhmnr/empyrean/network/members
[stars-shield]: https://img.shields.io/github/stars/dhmnr/empyrean.svg?style=for-the-badge
[stars-url]: https://github.com/dhmnr/empyrean/stargazers
[issues-shield]: https://img.shields.io/github/issues/dhmnr/empyrean.svg?style=for-the-badge
[issues-url]: https://github.com/dhmnr/empyrean/issues
[license-shield]: https://img.shields.io/github/license/dhmnr/empyrean.svg?style=for-the-badge
[license-url]: https://github.com/dhmnr/empyrean/blob/master/LICENSE
