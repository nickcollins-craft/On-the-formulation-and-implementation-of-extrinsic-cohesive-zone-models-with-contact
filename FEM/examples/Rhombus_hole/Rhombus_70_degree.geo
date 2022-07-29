cl__1 = 1;
/* Set up geometry for rhombus hole per Doitrand et al. All dimensions in mm. With
σ_c = 80 MPa, G_c = 0.25N/mm, δ_c = 0.00625*/
l_c = 0.002;
Point(1) = {0, 4.95, 0, l_c};
Point(2) = {4.95*Tan(35*Pi/180), 0, 0, 2.0};
Point(3) = {20.0, 0.0, 0, 2.0};
Point(4) = {20.0, 30.0, 0, 2.0};
Point(5) = {0, 30.0, 0, 0.5};
Point(6) = {0, 5.95, 0, l_c};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(7) = {1, 2, 3, 4, 5, 6};
Plane Surface(8) = {7};
Physical Line("Dirichlet BC") = {2};
Physical Line("Contact BC") = {5, 6};
Physical Line("Free boundary") = {1, 3};
Physical Line("Applied displacement") = {4};
Physical Surface("Bulk material") = {8};
Recombine Surface{8};
Mesh.RecombinationAlgorithm = 2;
