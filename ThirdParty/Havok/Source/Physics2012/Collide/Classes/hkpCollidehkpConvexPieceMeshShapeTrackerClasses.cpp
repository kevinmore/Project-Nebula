/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpConvexPieceMeshShape";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpConvexPieceMeshShapeRegister() {}

#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceMeshShape.h>


// hkpConvexPieceMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Stats)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceMeshShape)
    HK_TRACKER_MEMBER(hkpConvexPieceMeshShape, m_convexPieceStream, 0, "hkpConvexPieceStreamData*") // const class hkpConvexPieceStreamData*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshShape, m_displayMesh, 0, "hkpShapeCollection*") // const class hkpShapeCollection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexPieceMeshShape, s_libraryName, hkpShapeCollection)


// Stats hkpConvexPieceMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexPieceMeshShape, Stats, s_libraryName)

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
