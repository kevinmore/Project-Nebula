/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DESTRUCTION_SHAPE_PROPERTIES_H
#define HKNP_DESTRUCTION_SHAPE_PROPERTIES_H

/// Helper class to associate a set of Destruction-specific properties with a physics shape in the filter pipeline.
/// It is used when a shape is exported without an owning rigid body.
class hknpDestructionShapeProperties : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);
		HK_DECLARE_REFLECTION();

	public:

		/// Constructor
		hknpDestructionShapeProperties();

		/// Serialization constructor
		hknpDestructionShapeProperties(hkFinishLoadedObjectFlag flag);

		/// Destructor
		virtual ~hknpDestructionShapeProperties();

	public:

		hkTransform m_worldFromShape;		///< Transforms from world to shape local space
		hkBool m_isHierarchicalCompound;	///< If true, the physics shape is a hierarchical compound shape.
		hkBool m_hasDestructionShapes;		///< Set to true if the compound shape includes only children with a hkdShape attribute group
};

#endif // HKNP_DESTRUCTION_SHAPE_PROPERTIES_H

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
