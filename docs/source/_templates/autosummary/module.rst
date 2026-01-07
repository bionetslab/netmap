{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

{# 1. Generate the module's own documentation #}
.. automodule:: {{ fullname }}

{# 2. Look for functions INSIDE this file and give them pages #}
{% block functions %}
{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:                          {# This line creates the function files #}
   :template: autosummary/function.rst
   :nosignatures:

{% for item in functions %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{# 3. Look for classes INSIDE this file and give them pages #}
{% block classes %}
{% if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:
   :template: autosummary/class.rst
   :nosignatures:

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{# 4. Look for sub-packages/sub-modules if this is a directory #}
{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
   :template: autosummary/module.rst

{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}