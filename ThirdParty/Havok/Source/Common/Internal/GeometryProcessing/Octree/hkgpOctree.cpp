/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Internal/hkInternal.h>
#include <Common/Internal/GeometryProcessing/Octree/hkgpOctree.h>

const int	hkgpOctree::lut_tetras[6][4] = {{0,3,1,7}, {0,5,7,1}, {0,5,4,7}, {0,6,2,7}, {0,7,4,6}, {0,7,2,3}};
const int	hkgpOctree::lut_edges[12][2] = {{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
const int	hkgpOctree::lut_faces[6][4] = {{0,2,4,6},{1,3,5,7},{2,3,6,7},{0,1,4,5},{0,1,2,3},{4,5,6,7}};
const int	hkgpOctree::lut_edge_volumes[3][8][2] = {{{0,1},{1,0},{0,3},{1,2},{0,5},{1,4},{0,7},{1,6}},
													{{0,2},{0,3},{1,0},{1,1},{0,6},{0,7},{1,4},{1,5}},
													{{0,4},{0,5},{0,6},{0,7},{1,0},{1,1},{1,2},{1,3}}};
const int	hkgpOctree::lut_face_volumes[3][8][2] = {{{0,6},{0,7},{1,4},{1,5},{2,2},{2,3},{3,0},{3,1}},
													{{0,5},{1,4},{0,7},{1,6},{2,1},{3,0},{2,3},{3,2}},
													{{0,3},{1,2},{2,1},{3,0},{0,7},{1,6},{2,5},{3,4}}};

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
