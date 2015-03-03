/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_GEOMCONVEXHULLTOLERANCES_H
#define HK_GEOMCONVEXHULLTOLERANCES_H

	///
	/// Tolerances used to define the accuracy of the hull.
	///
	/// NOTE: These tolerances are internal, and are not settable by the user
struct hkGeomConvexHullTolerances
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomConvexHullTolerances );

		/// Specifies whether or not to run the postFilter.
	hkBool m_postFilter;

		/// Specifies whether or not to scale the vertices to fit in a unit cube.
	hkBool m_runConvertToUnitCube;

		/// Specifies the mode to run the convexHullBuilder in.
		/// This option is specified in the interface to the convex hull builder.
	hkBool m_accurateButSlow;

		/// The set of input vertices are filtered to remove coincident vertices. This means that
		/// if a vertex is found that is within weld_tolerance of a previous vertex,
		/// then it and all vertices afterwards within that tolerance are removed from the list.
	hkReal m_weld_tolerance;

		/// Identifies collinear triangles using a degenerate_tolerance (where area of triangle < degenerate_tolerance).
		/// For each degenerate triangle found, the interior point is removed.
		///
		/// This tolerance has the most effect on the accuracy of the hull - change it's value between 4e-6 and 2e-6
		/// to tweak hulls that don't work properly.
	hkReal m_degenerate_tolerance;


		/// Remove vertices which are redundant due to coplanarity. For each vertex, all adjacent faces are examined.
		/// If all the normals are all within an angle determined by coplanar_vertices_tolerance of the vertex normal
		/// (determined by the average of all face normals), the vertex is removed.
	hkReal m_coplanar_vertices_tolerance;

		/// "Welds" coplanar triangles, so that only one plane is generated for a set of coplanar triangles.
		/// It uses a coplanar_plane_tolerance, which is used to compare the distance of each vertex from the closest plane.
		/// This tolerance is multiplied by the total length of the shape.
	hkReal m_coplanar_plane_tolerance;
	
		/// No points in the point cloud lie outside any triangle to within a tolerance, called the coplanar_tolerance.
		/// This is determined by testing all points against the all planes.
	hkReal m_coplanar_tolerance;
	
		/// There are two plane equations that share a edge and are nearly exactly opposite.
		/// This is determined by adding the plane equations and checking if the length is less than this tolerance.
		/// This does allow for very long, thin wedge-shaped objects to be considered planar,
		/// but since the distinction is only used when testing the validity of the hull it doesn't really matter.
	hkReal m_oppositeNormal_tolerance;


	//
	// Internal tolerances
	//

	hkReal m__min_proj;
	hkReal m__maxAngle;
	hkReal m__planeEqnMinLength;
	hkReal m__tol;
	hkReal m__tol2;
	hkReal m__tol3;


	hkGeomConvexHullTolerances() :
		m_postFilter(true),
		m_weld_tolerance(2e-5f),
		m_degenerate_tolerance(4e-6f),
		m_coplanar_vertices_tolerance(1e-6f),  //1e-4f
		m_coplanar_plane_tolerance(1e-5f),
		m_coplanar_tolerance(0.05f),
		m_oppositeNormal_tolerance(1e-6f),

		m__min_proj(1e-6f),
		m__maxAngle(1e-8f),
		m__planeEqnMinLength(1e-6f),
		m__tol(10e-5f),
		m__tol2(10e-6f),
		m__tol3(2e-5f)

	{
	}
};

#endif //HK_GEOMCONVEXHULLTOLERANCES_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
