/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

int geometryUtils_main()
{
	{
		// Essentially points in a line at values 0, 0.9, 1.8, 2.7, 3.6, 4.0, 4.5
		// To with tolerance 1.0 these collapse as follows:
		// 0 and 0.9->0
		// 1.8 and 2.7->1.8
		// 3.6, 4.0 and 4.5 -> 3.6

		hkGeometry geom;
		geom.m_vertices.setSize(8);

		hkVector4* verts = geom.m_vertices.begin();
		verts[0].set( 0, 0,  0);
		verts[1].set( 0.9f, 0,  0);
		verts[2].set( 1.8f, 0,  0);
		verts[3].set( 2.7f, 0,  0);
		verts[4].set( 3.6f,  0,  0);
		verts[5].set( 4.0f,  0,  0);
		verts[6].set( 4.5f,  0,  0);
		verts[7].set( 10,  10,  0);

		geom.m_triangles.setSize(6);
		hkGeometry::Triangle* tris = geom.m_triangles.begin();
		tris[0].m_a = 0; tris[0].m_b = 1; tris[0].m_c = 6;
		tris[1].m_a = 0; tris[1].m_b = 2; tris[1].m_c = 6;	
		tris[2].m_a = 0; tris[2].m_b = 3; tris[2].m_c = 6;
		tris[3].m_a = 0; tris[3].m_b = 4; tris[3].m_c = 6;
		tris[4].m_a = 0; tris[4].m_b = 5; tris[4].m_c = 6;	
		tris[5].m_a = 0; tris[5].m_b = 6; tris[5].m_c = 6;	

		hkArray<int> vertexRemap;
		hkArray<int> triangleRemap;
		hkGeometryUtils::weldVertices( geom, 1.0f);

		HK_TEST( geom.m_vertices.getSize() == 4 );	
	}

	// Create a geometry with duplicate verts
	{
		hkGeometry geom;
		geom.m_vertices.setSize(6);

		hkVector4* verts = geom.m_vertices.begin();
		verts[0].set(-1,  1,  0);
		verts[1].set(-1, -1,  0);
		verts[2].set( 1, -1,  0);
		verts[3].set( 1, -1,  0);
		verts[4].set( 1,  1,  0);
		verts[5].set(-1,  1,  0);

		geom.m_triangles.setSize(2);
		hkGeometry::Triangle* tris = geom.m_triangles.begin();
		tris[0].m_a = 0; tris[0].m_b = 1; tris[0].m_c = 2;
		tris[1].m_a = 3; tris[1].m_b = 4; tris[1].m_c = 5;

		hkArray<int> vertexRemap;
		hkArray<int> triangleRemap;
		hkGeometryUtils::weldVertices( geom, 0.0f, false, vertexRemap, triangleRemap );

		HK_TEST( geom.m_vertices.getSize() == 4 );
		HK_TEST( geom.m_triangles.getSize() == 2 );
		HK_TEST( vertexRemap[2] == vertexRemap[3] );
		HK_TEST( vertexRemap[0] == vertexRemap[5] );
		HK_TEST( triangleRemap[0] == 0 );
		HK_TEST( triangleRemap[1] == 1 );
	}
	
	// Create a geometry with vertices which will be welded to create a degenerate triangle
	{
		hkGeometry geom;
		geom.m_vertices.setSize(6);

		hkVector4* verts = geom.m_vertices.begin();
		verts[0].set( 1,  1,  0);
		verts[1].set( 1, -1,  0);
		verts[2].set(-1, -1,  0);
		verts[3].set(-1, -1,  0);
		verts[4].set( 0.9f,  1,  0);
		verts[5].set( 1,  1,  0);

		geom.m_triangles.setSize(2);
		hkGeometry::Triangle* tris = geom.m_triangles.begin();
		tris[0].m_a = 0; tris[0].m_b = 1; tris[0].m_c = 2;
		tris[1].m_a = 3; tris[1].m_b = 4; tris[1].m_c = 5;

		hkArray<int> vertexRemap;
		hkArray<int> triangleRemap;
		hkGeometryUtils::weldVertices( geom, 0.2f, true, vertexRemap, triangleRemap );

		HK_TEST( geom.m_vertices.getSize() == 3 );
		HK_TEST( geom.m_triangles.getSize() == 1 );
		HK_TEST( vertexRemap[2] == vertexRemap[3] );
		HK_TEST( vertexRemap[0] == vertexRemap[5] );
		HK_TEST( vertexRemap[4] == vertexRemap[5] );
		HK_TEST( triangleRemap[0] ==  0 );
		HK_TEST( triangleRemap[1] == -1 );
	}

	{
		hkPseudoRandomGenerator rand(123);
		hkGeometry randGeom;
		randGeom.m_vertices.setSize( 50 );
		for (int v=0 ; v < randGeom.m_vertices.getSize(); v++)
		{
			rand.getRandomVector11( randGeom.m_vertices[v] );
		}

		const hkBool reorder = rand.getRandChar(2) != 0;

		hkGeometry original = randGeom;

		hkReal tolerance = 0.33f;
		hkArray<int> remapVerts;
		hkArray<int> remapTri;
		hkGeometryUtils::weldVertices( randGeom, tolerance, reorder, remapVerts, remapTri );

		for (int v=0 ; v < randGeom.m_vertices.getSize()-1; v++)
		{
			for (int k=v+1 ; k < randGeom.m_vertices.getSize(); k++)
			{
				HK_TEST( randGeom.m_vertices[v].distanceToSquared( randGeom.m_vertices[k] ).getReal() > tolerance * tolerance );
			}
		}

		// Check all collapsed vertices are within tolerance
		for (int v=0 ; v < original.m_vertices.getSize(); v++)
		{
			int remap = remapVerts[v];
			HK_TEST( original.m_vertices[v].distanceToSquared( randGeom.m_vertices[ remap ] ).getReal() <= tolerance * tolerance );
		}
	}

	{
		// Check removeDuplicateTriangles
		hkGeometry geometry;
		hkTransform t; t.setIdentity();
		hkVector4 top; top.set(10,5,7);
		hkVector4 bottom; bottom.set(-10,-7,-3);
		float radius = 4.0;
		int N = 100;
		hkGeometryUtils::createCapsuleGeometry(top, bottom, radius, N, N, t, geometry);

		// Make a geometry with about 10% of the triangles duplicated (with indices possibly cycled and winding flipped).
		int numDuplicates = geometry.m_triangles.getSize() / 10;

		int rotate = 0;
		int numIdentical = 0;
		hkArray<hkGeometry::Triangle>::Temp duplicates;
		hkArray<int>::Temp seenBefore;

		for (int i=0; i<numDuplicates; ++i)
		{
			int ti = int(geometry.m_triangles.getSize() * hkUnitTest::rand01());
			hkGeometry::Triangle duplicate = geometry.m_triangles[ti];

			// Assuming no degenerate ones
			HK_TEST((duplicate.m_a != duplicate.m_b) && (duplicate.m_b != duplicate.m_c) && (duplicate.m_c != duplicate.m_a));

			if (seenBefore.indexOf(ti) != -1)
			{
				// Some triangles may be chosen by the random number generator more than once. 
				// Make those have identical indices otherwise hard to keep track of how many identical ones there are ...
				numIdentical++;
			}
			else
			{
				int indices[3] = {duplicate.m_a, duplicate.m_b, duplicate.m_c};
				int newindices[3];
				newindices[0] = indices[(0+rotate) % 3];
				newindices[1] = indices[(1+rotate) % 3];
				newindices[2] = indices[(2+rotate) % 3];

				if (rotate%7)
				{
					duplicate.m_a  = newindices[0];
					duplicate.m_b  = newindices[1];
					duplicate.m_c  = newindices[2];
					if ( (rotate%3) == 0 ) numIdentical++;
				}
				else
				{
					// Every seventh has different winding
					duplicate.m_a  = newindices[2];
					duplicate.m_b  = newindices[1];
					duplicate.m_c  = newindices[0];
				}
			}
		
			duplicates.pushBack(duplicate);
			seenBefore.pushBack(ti);
			rotate++;
		}

		for (int ti=0; ti<duplicates.getSize(); ++ti)
		{
			int newti = int(geometry.m_triangles.getSize() * hkUnitTest::rand01());
			geometry.m_triangles.insertAt(newti, duplicates[ti]);
		}

		
		hkGeometry geometryWithDuplicates = geometry;

		// Check version which ignores winding
		{
			hkGeometry copyGeom = geometry;

			hkArray<int> triangleMapOut;
			bool ignoreWinding = true;
			hkGeometryUtils::removeDuplicateTriangles(copyGeom, triangleMapOut, ignoreWinding);
			hkGeometry* geometryWithoutDuplicates = &copyGeom;

			int numTrisBefore = geometryWithDuplicates.m_triangles.getSize();
			int numTrisAfter = geometryWithoutDuplicates->m_triangles.getSize();

			// Check the right number of triangles were removed
			HK_TEST( numTrisBefore - numTrisAfter == numDuplicates );

			// Check that map correctly maps original triangles to duplicates in the output geometry, identical up to winding
			for (int ti=0; ti<triangleMapOut.getSize(); ++ti)
			{
				int origTriIndex = ti;
				int duplicateTriIndex = triangleMapOut[ti];
				hkGeometry::Triangle origTri = geometryWithDuplicates.m_triangles[origTriIndex];
				hkGeometry::Triangle duplicateTri = geometryWithoutDuplicates->m_triangles[duplicateTriIndex];

				// Bubble sort!
				if (origTri.m_a > origTri.m_b)	{ hkAlgorithm::swap( origTri.m_a, origTri.m_b ); }
				if (origTri.m_a > origTri.m_c)	{ hkAlgorithm::swap( origTri.m_a, origTri.m_c ); }
				if (origTri.m_b > origTri.m_c)	{ hkAlgorithm::swap( origTri.m_b, origTri.m_c ); }
				if (duplicateTri.m_a > duplicateTri.m_b)	{ hkAlgorithm::swap( duplicateTri.m_a, duplicateTri.m_b ); }
				if (duplicateTri.m_a > duplicateTri.m_c)	{ hkAlgorithm::swap( duplicateTri.m_a, duplicateTri.m_c ); }
				if (duplicateTri.m_b > duplicateTri.m_c)	{ hkAlgorithm::swap( duplicateTri.m_b, duplicateTri.m_c ); }

				HK_TEST(origTri.m_a == duplicateTri.m_a);
				HK_TEST(origTri.m_b == duplicateTri.m_b);
				HK_TEST(origTri.m_c == duplicateTri.m_c);
			}
		}

		// Check version which respects winding
		{
			hkGeometry copyGeom = geometry;

			hkArray<int> triangleMapOut;
			bool ignoreWinding = false;
			hkGeometryUtils::removeDuplicateTriangles(copyGeom, triangleMapOut, ignoreWinding);
			hkGeometry* geometryWithoutDuplicates = &copyGeom;

			int numTrisBefore = geometryWithDuplicates.m_triangles.getSize();
			int numTrisAfter = geometryWithoutDuplicates->m_triangles.getSize();

			// Check the right number of triangles were removed
			HK_TEST( numTrisBefore - numTrisAfter == numIdentical );

			// Check that map correctly maps original triangles to EXACT duplicates in the output geometry
			for (int ti=0; ti<triangleMapOut.getSize(); ++ti)
			{
				int origTriIndex = ti;
				int duplicateTriIndex = triangleMapOut[ti];
				hkGeometry::Triangle origTri = geometryWithDuplicates.m_triangles[origTriIndex];
				hkGeometry::Triangle duplicateTri = geometryWithoutDuplicates->m_triangles[duplicateTriIndex];
				HK_TEST(origTri.m_a == duplicateTri.m_a);
				HK_TEST(origTri.m_b == duplicateTri.m_b);
				HK_TEST(origTri.m_c == duplicateTri.m_c);
			}
		}
	}

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(geometryUtils_main, "Fast", "Common/Test/UnitTest/GeometryUtilities/", __FILE__     );

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
