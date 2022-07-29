cl__1 = 1;
/* Set up geometry for rhombus hole per Doitrand et al. All dimensions in mm. With
σ_c = 80 MPa, G_c = 0.25N/mm, δ_c = 0.00625*/
l_c = 0.002;
Point(1) = {0, 4.95, 0, l_c};
Point(2) = {4.95*Tan(50*Pi/180), 0, 0, 2.0};
Point(3) = {20.0, 0.0, 0, 2.0};
Point(4) = {20.0, 30.0, 0, 2.0};
Point(5) = {0, 30.0, 0, 2.0};
Point(6) = {0, 6.95, 0, l_c};
Point(7) = {0, 5.95, 0, 3*l_c};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};
Line Loop(8) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(9) = {8};
Physical Line("Dirichlet BC") = {2};
Physical Line("Contact BC") = {5, 6, 7};
Physical Line("Free boundary") = {1, 3};
Physical Line("Applied displacement") = {4};
Physical Surface("Bulk material") = {9};
Recombine Surface{9};
Mesh.RecombinationAlgorithm = 2;
