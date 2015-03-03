/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Geometry/hkGeometryUtil.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>

hkGeometryUtil::GridInput::GridInput(int numVertsX, hkVector4Parameter up)
{
	m_stepX.set(1,0,0);
	m_stepY.setCross( m_stepX, up);
	m_numVertsX = numVertsX;
	m_numVertsY = numVertsX;

	hkReal x = -0.5f * (numVertsX-1);
	hkReal y = -0.5f * (numVertsX-1);

	m_origin.set( x, 0,0 );

	hkSimdReal ySr; ySr.setFromFloat( y );
	m_origin.addMul( ySr, m_stepY );

}

void hkGeometryUtil::createGrid( const GridInput& input, hkGeometry* out, int defaultMaterial )
{
	hkVector4 posX = input.m_origin;
	int iB = out->m_vertices.getSize();
	out->m_vertices.reserve( out->m_vertices.getSize() + input.m_numVertsX * input.m_numVertsY );
	out->m_triangles.reserve( out->m_triangles.getSize() + 2 * (input.m_numVertsX-1) * (input.m_numVertsY-1) );
	for ( int x=0; x<input.m_numVertsX; x++ )
	{
		hkVector4 posY = posX;
		for ( int y=0; y<input.m_numVertsY; y++ )
		{
			out->m_vertices.expandByUnchecked(1)[0] = posY;
			if ( x>0 && y >0)
			{
				int yf = input.m_numVertsY;
				int i = iB + y + x*yf;
				out->m_triangles.expandByUnchecked(1)->set( i   -0, i -1, i-yf-1, defaultMaterial );	
				out->m_triangles.expandByUnchecked(1)->set( i-yf-1, i-yf, i   -0, defaultMaterial );
			}
			posY.add(input.m_stepY);
		}
		posX.add(input.m_stepX);
	}
}

void hkGeometryUtil::createSphere(hkVector4Parameter center, hkReal radius, int numSteps, hkGeometry* geometryOut, 
								  int material)
{
	HK_ASSERT(0x737893e6, geometryOut != HK_NULL);
	hkArray<hkVector4>::Temp vertices;
	vertices.reserve(numSteps * numSteps);

	// Generate sphere vertices with uniform sampling
	hkSimdReal radiusSimd; radiusSimd.setFromFloat(radius);
	const hkReal invSteps = 1.0f / (numSteps - 1);		
	for (int x = 0; x < numSteps; ++x)
	{
		for (int y = 0; y < numSteps; ++y)
		{
			hkVector4 uv; uv.set(x * invSteps, y * invSteps, 0.0f);
			hkVector4& vertex = vertices.expandOne(); 
			hkGeometryProcessing::octahedronToNormal(uv, vertex);
			vertex.mul(radiusSimd);
			vertex.add(center);
		}
	}

	// Create convex hull from the vertices and use it to generate geometry
	hkgpConvexHull hull;
	hull.build(vertices);
	hull.generateGeometry(hkgpConvexHull::SOURCE_VERTICES, *geometryOut, material);
}

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
