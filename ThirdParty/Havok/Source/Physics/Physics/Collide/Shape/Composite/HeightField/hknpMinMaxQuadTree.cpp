/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpMinMaxQuadTree.h>


void hknpMinMaxQuadTree::init(int sizeX, int sizeZ, hkSimdRealParameter minHeight, hkSimdRealParameter maxHeight)
{
	HK_ASSERT2(0xfefe012c, !(sizeX&1) && !(sizeZ&1), "hknpMinMaxQuadTree must have an even size");
	m_coarseTreeData.clear();
	// we store 2 x 2 heights in one entry so divide by two
	sizeX = sizeX/2;
	sizeZ = sizeZ/2;
	unsigned int s = hkMath::max2(sizeX, sizeZ)-1;
	int levels = 1;
	while ( s )
	{
		s = s>>1;
		levels++;
	}
	m_coarseTreeData.reserveExactly(levels);

	for (int i=0; i<levels; i++)
	{
		HK_ASSERT(0xbabadfdf, (i==levels-1) || (sizeX!=1 || sizeZ!=1));
		HK_ASSERT(0xbabadedf, (i!=levels-1) || (sizeX==1 && sizeZ==1));
		MinMaxLevel& level = m_coarseTreeData.expandOne();
		level.m_xRes = (hkUint16)sizeX;
		level.m_zRes = (hkUint16)sizeZ;
		level.m_minMaxData.setSize(4*sizeX*sizeZ);
		sizeX = (sizeX+1)/2;
		sizeZ = (sizeZ+1)/2;
	}

	hkSimdReal min = minHeight-hkSimdReal_Eps;
	hkSimdReal max = maxHeight+hkSimdReal_Eps;
	m_offset.setAll(min);
	m_multiplier = ((max-min)*hkSimdReal::fromFloat(1.0f/0xffff)).getReal();
	m_invMultiplier = hkReal(1)/m_multiplier;
}

	// Helper functions to buildCoarseTree
HK_FORCE_INLINE static hkUint16 hknpMinMaxQuadTree_calcNextMinHeight(hknpMinMaxQuadTree* self, int level, int x, int z)
{
	HK_ASSERT(0x42893572, level<self->m_coarseTreeData.getSize());
	const hknpMinMaxQuadTree::MinMaxLevel& coarseLevel = self->m_coarseTreeData[level];
	if (x>=coarseLevel.m_xRes || z>=coarseLevel.m_zRes)
	{
		return 0xffff;
	}
	int index = 4*(x*coarseLevel.m_zRes+z);
	hkUint32 min1 = coarseLevel.m_minMaxData[index];
	hkUint32 min2 = coarseLevel.m_minMaxData[index+1];
	return (hkUint16)hkMath::min2(hkMath::min2(min1>>16, min1&0xffff), hkMath::min2(min2>>16, min2&0xffff));
}

HK_FORCE_INLINE static hkUint16 hknpMinMaxQuadTree_calcNextMaxHeight(hknpMinMaxQuadTree* self, int level, int x, int z)
{
	HK_ASSERT(0x42893572, level<self->m_coarseTreeData.getSize());
	const hknpMinMaxQuadTree::MinMaxLevel& coarseLevel = self->m_coarseTreeData[level];
	if (x>=coarseLevel.m_xRes || z>=coarseLevel.m_zRes)
	{
		return 0;
	}
	int index = 4*(x*coarseLevel.m_zRes+z);
	hkUint32 max1 = coarseLevel.m_minMaxData[index+2];
	hkUint32 max2 = coarseLevel.m_minMaxData[index+3];
	return (hkUint16)hkMath::max2(hkMath::max2(max1>>16, max1&0xffff), hkMath::max2(max2>>16, max2&0xffff));
}

void hknpMinMaxQuadTree::updateRegion(int x0, int z0, int x1, int z1, hkReal* newMinOut, hkReal* newMaxOut)
{
	for (int level=1; level<m_coarseTreeData.getSize(); level++)
	{
		MinMaxLevel& coarseLevel = m_coarseTreeData[level];
		x0 = x0/2; x1 = x1/2; z0 = z0/2; z1 = z1/2;
		for (int ix=x0; ix<=x1; ix++)
		{
			for (int iz=z0; iz<=z1; iz++)
			{
				hkUint16 vmin[4] = {
					hknpMinMaxQuadTree_calcNextMinHeight(this, level-1, 2*ix  , 2*iz  ),
					hknpMinMaxQuadTree_calcNextMinHeight(this, level-1, 2*ix+1, 2*iz  ),
					hknpMinMaxQuadTree_calcNextMinHeight(this, level-1, 2*ix+1, 2*iz+1),
					hknpMinMaxQuadTree_calcNextMinHeight(this, level-1, 2*ix  , 2*iz+1)
				};
				hkUint16 vmax[4] =  {
					hknpMinMaxQuadTree_calcNextMaxHeight(this, level-1, 2*ix  , 2*iz  ),
					hknpMinMaxQuadTree_calcNextMaxHeight(this, level-1, 2*ix+1, 2*iz  ),
					hknpMinMaxQuadTree_calcNextMaxHeight(this, level-1, 2*ix+1, 2*iz+1),
					hknpMinMaxQuadTree_calcNextMaxHeight(this, level-1, 2*ix  , 2*iz+1)
				};
				int index = 4*(ix*coarseLevel.m_zRes+iz);
				coarseLevel.m_minMaxData[index] = vmin[0] | (vmin[1]<<16);
				coarseLevel.m_minMaxData[index+1] = vmin[2] | (vmin[3]<<16);
				coarseLevel.m_minMaxData[index+2] = vmax[0] | (vmax[1]<<16);
				coarseLevel.m_minMaxData[index+3] = vmax[2] | (vmax[3]<<16);
			}
		}
	}
	hkVector4 qmin; qmin.setZero();
	hkVector4 qmax; qmax.setZero();
	getMinMax( m_coarseTreeData.getSize()-1, 0, 0, &qmin, &qmax );
	*newMinOut =  qmin.horizontalMin<4>().getReal();
	*newMaxOut =  qmax.horizontalMax<4>().getReal();
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
