/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_QUERY_AABB_NMP_UTIL_H
#define HKNP_QUERY_AABB_NMP_UTIL_H

class hknpQueryAabbNmp;


/// hknpQueryAabbNmp acts as an optimization to the shape-level AABB query.
/// It is directly tied to the shape the query is being performed on, i.e. a hknpQueryAabbNmp object cannot be shared
/// between different shapes.
/// It allows for testing if the results of a previous call to hknpShapeQueryInterface::queryAabb() on a specific shape
/// can be re-used or if the query has to be performed anew.
class hknpQueryAabbNmpUtil
{
	public:

		/// A buffer, sized such that it can hold an hknpQueryAabbNmp object.
		struct Buffer
		{
			enum
			{
				QUERY_AABB_NMP_BUFFER_SIZE = 2 * sizeof(hkVector4)
			};

			HK_ALIGN16( hkUint8 m_buffer[QUERY_AABB_NMP_BUFFER_SIZE] );
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpQueryAabbNmpUtil);

		/// Creates a new hknpQueryAabbNmp object by initializing the provided buffer.
		/// Also calls resetNmp() on the newly created hknpQueryAabbNmp.
		/// Returns the provided buffer, cast to hknpQueryAabbNmp.
		static hknpQueryAabbNmp* HK_CALL createNmp(Buffer *queryAabbNmpBuffer);

		/// Resets the hknpQueryAabbNmp.
		/// This method needs to be called prior to a call to hknpShape::queryAabbImpl().
		static void HK_CALL resetNmp(hknpQueryAabbNmp* nmp);

		/// Returns <true> if the hknpQueryAabbNmp is still valid and the results of a previous call to
		/// hknpShapeQueryInterface::queryAabb() can be re-used.
		static bool HK_CALL checkNmpStillValid(
			const hkAabb& aabb, const hknpQueryAabbNmp& nmp, hkUint8* HK_RESTRICT nmpTimeToLive);

		/// Internal helper method.
		static hkBool32 HK_CALL checkOverlapWithNmp(
			const hkAabb& queryAabb, const hkAabb& targetAabb, hknpQueryAabbNmp* nmpInOut);

	protected:

		friend class hknpBridge_hkpMeshShape;
		friend struct hknpCompositeCollisionCache;
		friend class hknpConvexCompositeCollisionDetector;
		friend class hknpCompositeCompositeCollisionDetector;

		/// Internal helper method.
		static void HK_CALL clearNmp(hknpQueryAabbNmp* nmp);

		/// Internal helper method.
		static hkSimdReal HK_CALL calcTIMWithNmp(const hkAabb& aabb, const hknpQueryAabbNmp& nmp);
};


#endif // HKNP_QUERY_AABB_NMP_UTIL_H

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
