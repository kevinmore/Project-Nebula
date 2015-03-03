/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_CD_BODY_H
#define HKNP_CD_BODY_H

#include <Physics/Physics/hknpTypes.h>


/// This structure combines a colliding body and its context.
struct hknpCdBodyBase
{
	HK_PAD_ON_SPU(const hknpBody*		) m_body;		///< The colliding body.
	HK_PAD_ON_SPU(const hknpBodyQuality*) m_quality;	///< The body's quality settings.
	HK_PAD_ON_SPU(const hknpMotion*		) m_motion;		///< The body's motion.
	HK_PAD_ON_SPU(const hknpShape*		) m_rootShape;	///< The body's root shape.
	HK_PAD_ON_SPU(const hknpMaterial*	) m_material;	///< The material, either taken from the body or overridden by a shape tag codec.
	HK_PAD_ON_SPU(hknpShapeKey			) m_shapeKey;	///< The leaf shape's shape key or HKNP_INVALID_SHAPE_KEY for a convex body.
};


/// This structure is an extension to hknpCdBodyBase and includes additional information on the colliding leaf shape.
struct hknpCdBody : public hknpCdBodyBase
{
	HK_PAD_ON_SPU(hkUint32				) m_collisionFilterInfo;	///< The collision filter info, either taken from the body or overridden by a shape tag codec.
	HK_PAD_ON_SPU(const hknpShape*		) m_leafShape;				///< The convex leaf shape. This can either be an actual leaf inside a hierarchy or the m_rootShape for a convex body.
	HK_PAD_ON_SPU(const hkTransform*	) m_transform;				///< The leaf shape's world transform.
};


#endif	//HKNP_CD_BODY_H

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
