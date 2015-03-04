/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppCompilerInput.h>

hkpMoppCompilerInput::hkpMoppCompilerInput()
{
	m_absoluteFitToleranceOfTriangles = 0.3f;
	m_relativeFitToleranceOfInternalNodes = 0.4f;
	m_absoluteFitToleranceOfInternalNodes = 0.1f;
	m_absoluteFitToleranceOfAxisAlignedTriangles.set( 0.05f, 0.05f, 0.05f );
	m_useShapeKeys = true;
	m_enablePrimitiveSplitting = true;
	m_enableInterleavedBuilding = true;
	m_cachePrimitiveExtents = false;
	m_enableChunkSubdivision = false;
}

void hkpMoppCompilerInput::setAbsoluteFitToleranceOfTriangles(float inTight)
{
	m_absoluteFitToleranceOfTriangles = inTight;
}

float hkpMoppCompilerInput::getAbsoluteFitToleranceOfTriangles() const
{
	return m_absoluteFitToleranceOfTriangles;
}

void hkpMoppCompilerInput::setAbsoluteFitToleranceOfAxisAlignedTriangles(const hkVector4& inTight)
{
	m_absoluteFitToleranceOfAxisAlignedTriangles = inTight;
}

hkVector4 hkpMoppCompilerInput::getAbsoluteFitToleranceOfAxisAlignedTriangles() const
{
	return m_absoluteFitToleranceOfAxisAlignedTriangles;
}

void hkpMoppCompilerInput::setRelativeFitToleranceOfInternalNodes(float inUnused)
{
	m_relativeFitToleranceOfInternalNodes = inUnused;
}

float hkpMoppCompilerInput::getRelativeFitToleranceOfInternalNodes() const
{
	return m_relativeFitToleranceOfInternalNodes;
}

void hkpMoppCompilerInput::setAbsoluteFitToleranceOfInternalNodes(float inMin)
{
	m_absoluteFitToleranceOfInternalNodes = inMin;
}

float hkpMoppCompilerInput::getAbsoluteFitToleranceOfInternalNodes() const
{
	return m_absoluteFitToleranceOfInternalNodes;
}

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
